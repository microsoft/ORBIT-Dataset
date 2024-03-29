"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file model.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/model.py) and
config_networks.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/config_networks.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

The original license is included below:

Copyright (c) 2019 John Bronskill, Jonathan Gordon, and James Requeima.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import numpy as np
import torch.nn as nn
from argparse import Namespace
from torch.nn.utils.stateless import functional_call

from data.utils import get_batch_indices
from model.feature_extractors import create_feature_extractor
from model.film import get_film_parameters, get_film_parameter_names, get_film_parameter_sizes, unfreeze_film
from model.feature_adapters import FilmParameterGenerator, NullGenerator
from model.poolers import MeanPooler
from model.set_encoders import SetEncoder, NullSetEncoder
from model.classifier_heads import LinearClassifier, VersaClassifier, PrototypicalClassifier, MahalanobisClassifier
from utils.optim import init_optimizer

class FewShotRecogniser(nn.Module):
    """
    Generic few-shot classification model.
    """
    def __init__(self, feature_extractor_name: str, adapt_features: bool, classifier: str, clip_length: int, batch_size: int, learn_extractor: bool, logit_scale: float=1.0):
        """
        Creates instance of FewShotRecogniser.
        """
        super(FewShotRecogniser, self).__init__()

        self.adapt_features = adapt_features
        self.learn_extractor = learn_extractor
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.logit_scale = logit_scale

        # configure feature extractor
        self.feature_extractor, self.film_parameter_names = create_feature_extractor(
            feature_extractor_name = feature_extractor_name,
            pretrained=True,
            with_film=self.adapt_features,
            learn_extractor=self.learn_extractor
        )

        # configure classifier
        self.classifier_name = classifier
        if classifier == 'linear':
            # classifier head will instead be appended per-task during train/test
            self.classifier = LinearClassifier(self.feature_extractor.output_size, self.logit_scale)
        elif classifier == 'versa':
            self.classifier = VersaClassifier(self.feature_extractor.output_size, self.logit_scale)
        elif classifier == 'proto':
            self.classifier = PrototypicalClassifier(self.logit_scale)
        elif classifier == 'proto_cosine':
            self.classifier = PrototypicalClassifier(self.logit_scale, distance_fn='cosine')
        elif classifier == 'mahalanobis':
            self.classifier = MahalanobisClassifier(self.logit_scale)
        else:
            raise ValueError(f"Classifier {classifier} not valid.")

        # configure frame pooler
        self.frame_pooler = MeanPooler(T=self.clip_length)

    def _set_device(self, device):
        self.device = device

    def _send_to_device(self):
        """
        Function that moves whole model to self.device.
        :return: Nothing.
        """
        self.to(self.device)
    
    def _get_features(self, clips, film_dict={}, ops_counter=None):
        """
        Function that passes clips through an adapted feature extractor to get adapted (and flattened) frame features.
        :param clips: (torch.Tensor) Tensor of clips, each composed of self.clip_length contiguous frames.
        :param film_dict: (dict) Generated FiLM parameters. Empty dict if adapt_features=False, or finetuning.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (torch.Tensor) Adapted frame features flattened across all clips.
        """

        if len(clips.shape) == 5:
            num_clips, clip_length, c, h, w = clips.shape
            clips = clips.reshape(num_clips*clip_length, c, h, w)

        clips = clips.to(self.device, non_blocking=True)
       
        if film_dict: # if film parameters have been generated, use stateless call
            features = functional_call(self.feature_extractor, film_dict, clips, kwargs=None)
        else:
            features = self.feature_extractor(clips)

        if ops_counter:
            ops_counter.compute_macs(self.feature_extractor, clips)

        return features

    def _get_features_in_batches(self, clips, film_dict={}, ops_counter=None):
        """
        Function that passes clips in batches through an adapted feature extractor to get adapted (and flattened) frame features.
        :param clips: (torch.Tensor) Clips, each composed of self.clip_length contiguous frames.
        :param film_dict: (dict) Generated FiLM parameters. Empty dict if adapt_features=False, or finetuning.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (torch.Tensor) Adapted frame features flattened across all clips.
        """
        features = []

        num_clips = len(clips)
        num_batches = int(np.ceil(float(num_clips) / float(self.batch_size)))
        for batch in range(num_batches):
            batch_start_index, batch_end_index = get_batch_indices(batch, num_clips, self.batch_size)
            batch_clips = clips[batch_start_index:batch_end_index]
            if len(batch_clips.shape) == 5:
                batch_clips = batch_clips.flatten(end_dim=1)

            batch_clips = batch_clips.to(self.device, non_blocking=True)
            if film_dict: # if film parameters have been generated, use stateless call
                batch_features = functional_call(self.feature_extractor, film_dict, batch_clips, kwargs=None)
            else:
                batch_features = self.feature_extractor(batch_clips)

            if ops_counter:
                ops_counter.compute_macs(self.feature_extractor, batch_clips)

            features.append(batch_features)

        return torch.cat(features, dim=0)

    def _pool_features(self, features, ops_counter=None):
        """
        Function that pools frame features per clip.
        :param features: (torch.Tensor) Frame features i.e. flattened as (num_clips*self.clip_length) x (feat_dim).
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (torch.Tensor) Frame features pooled per clip i.e. as (num_clips) x (feat_dim).
        """
        pooled_features = self.frame_pooler(features)
        if ops_counter:
            ops_counter.add_macs(features.size(0) * features.size(1))

        return pooled_features

    def set_test_mode(self, test_mode):
        """
        Function that flags if model is being evaluated. Relevant for self._set_batch_norm_state().
        :param test_mode: (bool) True if model is being evaluated, otherwise True.
        :return: Nothing.
        """
        self.test_mode = test_mode
    
    def _set_batch_norm_state(self):
        """
        Function that sets batch norm modules to appropriate train() or eval() states.
        :return: Nothing.
        """
        self.eval()
        if self.learn_extractor and not self.test_mode: # if meta-training and extractor is unfrozen, then it must be in train() mode
            self.feature_extractor.train()

class MultiStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot classification model that is personalised in multiple forward-backward steps (e.g. MAML, FineTuner).
    """
    def __init__(self, feature_extractor_name: str, adapt_features: bool, classifier: str, clip_length: int, batch_size: int, learn_extractor: bool, logit_scale: float=1.0):
        """
        Creates instance of MultiStepFewShotRecogniser.
        """
        FewShotRecogniser.__init__(self, feature_extractor_name, adapt_features, classifier, clip_length, batch_size, learn_extractor, logit_scale)
        
        # configure film
        if self.adapt_features:
            self.film_parameter_sizes = get_film_parameter_sizes(self.film_parameter_names, self.feature_extractor)
            unfreeze_film(self.film_parameter_names, self.feature_extractor) # enable film grads - used only for finetuning
    
    def _reset(self):
        """
        Function that resets model's task-specific parameters after a task is processed.
        :return: Nothing.
        """
        self.classifier.reset()

    def personalise(self, context_clips, context_labels, learning_args, ops_counter=None):
        """
        Function that learns a new task by taking a fixed number of gradient steps on the task's full context set. For each task, a new linear classification layer is added (and FiLM layers if self.adapt_features == True).
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :param learning_args: (dict) Hyperparameters for personalisation.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        self._set_batch_norm_state()

        num_grad_steps = learning_args.pop('num_grad_steps')
        learning_rate = learning_args.pop('learning_rate')
        optimizer = learning_args.pop('optimizer')
        loss_fn = learning_args.pop('loss_fn')
        extractor_lr_scale = learning_args.pop('extractor_lr_scale')
        optimizer_kwargs = Namespace(**learning_args)

        num_classes = len(torch.unique(context_labels))
        self.init_classifier(num_classes)
        personalize_optimizer = init_optimizer(self, learning_rate, optimizer, optimizer_kwargs, extractor_lr_scale)

        batch_context_set_size = len(context_labels)
        num_batches = int(np.ceil(float(batch_context_set_size) / float(self.batch_size)))
        for _ in range(num_grad_steps):
            for batch in range(num_batches):
                batch_start_index, batch_end_index = get_batch_indices(batch, batch_context_set_size, self.batch_size)
                batch_context_clips = context_clips[batch_start_index:batch_end_index]
                batch_context_labels = context_labels[batch_start_index:batch_end_index]
                batch_len = len(context_labels[batch_start_index:batch_end_index])
               
                batch_context_features = self._get_features(batch_context_clips, ops_counter=ops_counter)
                batch_context_features = self._pool_features(batch_context_features, ops_counter=ops_counter)
                batch_context_logits = self.classifier.predict(batch_context_features, ops_counter=ops_counter)
                loss = loss_fn(batch_context_logits, batch_context_labels.to(self.device))
                loss *= batch_len/batch_context_set_size
                loss.backward()

            personalize_optimizer.step()
            personalize_optimizer.zero_grad()
    
    def predict(self, clips, ops_counter=None):
        """
        Function that processes target clips in batches to get logits over object classes for each clip.
        :param clips: (torch.Tensor) Clips, each composed of self.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (torch.Tensor) Logits over object classes for each clip in clips.
        """
        self._set_batch_norm_state()
        features = self._get_features_in_batches(clips, ops_counter=ops_counter)
        features = self._pool_features(features, ops_counter=ops_counter)
        return self.classifier.predict(features, ops_counter=ops_counter)

    def personalise_with_lite(self, context_clips, context_labels):
        NotImplementedError
    
    def init_classifier(self, num_classes:int):
        """
        Function that initialises classifier's parameters.
        :return: Nothing.
        """
        self.classifier.init(num_classes)
        self.classifier.to(self.device)

class SingleStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot classification model that is personalised in a single forward step (e.g. CNAPs, ProtoNets).
    """
    def __init__(self, feature_extractor_name: str, adapt_features: bool, classifier: str, clip_length: int, batch_size: int, learn_extractor: bool, num_lite_samples: int, logit_scale: float=1.0):
        """
        Creates instance of SingleStepFewShotRecogniser.
        """
        FewShotRecogniser.__init__(self, feature_extractor_name, adapt_features, classifier, clip_length, batch_size, learn_extractor, logit_scale)
        self.num_lite_samples = num_lite_samples
        
        # configure film generator
        if self.adapt_features:
            self.set_encoder = SetEncoder()
            self.film_parameter_sizes = get_film_parameter_sizes(self.film_parameter_names, self.feature_extractor)
            initial_film_parameters = get_film_parameters(self.film_parameter_names, self.feature_extractor)
            self.film_generator = FilmParameterGenerator(
                self.film_parameter_sizes,
                initial_film_parameters,
                pooled_size=self.set_encoder.output_size,
                hidden_size=self.set_encoder.output_size
            )
        else:
            self.set_encoder = NullSetEncoder()
            self.film_generator = NullGenerator()
    
    def _reset(self):
        """
        Function that resets model's task-specific parameters after a task is processed.
        :return: Nothing.
        """
        self.film_dict = None
        self.classifier.reset()

    def _clear_caches(self):
        """
        Function that clears caches if training with LITE.
        :return: Nothing.
        """
        self.reps_cache = None
        self.features_cache = None

    def personalise(self, context_clips, context_labels, ops_counter=None):
        """
        Function that learns a new task by performing a forward pass of the task's context set.
        :param context_clips: (torch.Tensor) Context clips each composed of self.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        self._set_batch_norm_state()
        task_embedding = self._get_task_embedding_in_batches(context_clips, ops_counter)
        self.film_dict = self._generate_film_params(task_embedding, ops_counter)
        context_features = self._get_features_in_batches(context_clips, self.film_dict, ops_counter)
        context_features = self._pool_features(context_features, ops_counter)
        self.classifier.configure(context_features, context_labels, ops_counter)

    def personalise_with_lite(self, context_clips, context_labels):
        """
        Function that learns a new task by performning a forward pass of the task's context set with LITE. Namely a random subset of the context set (self.num_lite_samples) is processed with back-propagation enabled, while the remainder is processed with back-propagation disabled.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :return: Nothing.
        """
        self._set_batch_norm_state()
        shuffled_idxs = np.random.permutation(len(context_clips))
        grad_idxs = shuffled_idxs[0:self.num_lite_samples]
        no_grad_idxs = shuffled_idxs[self.num_lite_samples:]
        task_embedding = self._get_task_embedding_with_split_batch(context_clips, grad_idxs, no_grad_idxs)
        self.film_dict = self._generate_film_params(task_embedding)
        context_features = self._get_features_with_split_batch(context_clips, self.film_dict, grad_idxs, no_grad_idxs)
        context_features = self._pool_features(context_features)
        self.classifier.configure(context_features, context_labels[shuffled_idxs])
    
    def _get_task_embedding(self, context_clips, ops_counter=None, aggregation='mean'):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding.
        :param context_clips: (torch.Tensor) Tensor of context clips, each composed of self.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param aggregation: (str) Method to aggregate clip encodings from self.set_encoder.
        :return: (torch.Tensor or None) Task embedding.
        """
        context_clips = context_clips.to(self.device, non_blocking=True)
        reps = self.set_encoder(context_clips)

        if ops_counter:
            ops_counter.compute_macs(self.set_encoder, context_clips)

        return self.set_encoder.aggregate(reps, aggregation=aggregation)

    def _get_task_embedding_in_batches(self, context_clips, ops_counter=None, aggregation='mean'):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param aggregation: (str) Method to aggregate clip encodings from self.set_encoder.
        :return: (torch.Tensor or None) Task embedding.
        """
        if isinstance(self.set_encoder, NullSetEncoder):
            return None
        
        reps = []
        num_clips = len(context_clips)
        num_batches = int(np.ceil(float(num_clips) / float(self.batch_size)))
        for batch in range(num_batches):
            batch_start_index, batch_end_index = get_batch_indices(batch, num_clips, self.batch_size)
            batch_clips = context_clips[batch_start_index:batch_end_index]
            batch_clips = batch_clips.to(self.device, non_blocking=True)
            batch_reps = self.set_encoder(batch_clips)

            if ops_counter:
                ops_counter.compute_macs(self.set_encoder, batch_clips)

            reps.append(batch_reps)

        return self.set_encoder.aggregate(reps, aggregation=aggregation)
    
    def _get_task_embedding_with_split_batch(self, context_clips, grad_idxs, no_grad_idxs):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding with LITE.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param grad_idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation enabled.
        :param no_grad_idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation disabled.
        :return: (torch.Tensor or None) Task embedding.
        """
        if isinstance(self.set_encoder, NullSetEncoder):
            return None

        self._set_batch_norm_state()
        # cache set encoder reps if they haven't been cached yet
        if self.reps_cache is None:
            with torch.set_grad_enabled(False):
                self.reps_cache = self._get_task_embedding_in_batches(context_clips, aggregation='none')

        # now select some random clips that will have gradients enabled and process those
        with torch.set_grad_enabled(True):
            reps_with_grads = self._get_task_embedding(context_clips[grad_idxs], aggregation='none')
      
        # now get reps for the rest of the clips which have grads disabled
        reps_without_grads = self.reps_cache[no_grad_idxs]

        # return mean pooled reps
        return torch.cat((reps_with_grads, reps_without_grads)).mean(dim=0)
    
    def _get_features_with_split_batch(self, context_clips, film_dict, grad_idxs, no_grad_idxs):
        """
        Function that gets adapted clip features for a task's context set with LITE.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param film_dict: (dict) Parameters of all FiLM layers. Empty dict if adapt_features=False.
        :param grad_idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation enabled.
        :param no_grad_idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation disabled.
        :return: (torch.Tensor) Adapted frame features per frame i.e. as (num_clips*clip_length) x (feat_dim).
        """
        self._set_batch_norm_state()
        if self.features_cache is None: 
            with torch.set_grad_enabled(False):
                self.features_cache = self._get_features_in_batches(context_clips, film_dict)

        # now select some random clips that will have gradients enabled and process those
        with torch.set_grad_enabled(True):
            features_with_grads = self._get_features(context_clips[grad_idxs], film_dict)
      
        # now get features for the rest of the clips which have grads disabled
        features_without_grads = self.features_cache[no_grad_idxs]

        # return all features
        return torch.cat((features_with_grads, features_without_grads))
    
    def _generate_film_params(self, task_embedding, ops_counter=None):
        """
        Function that processes a task embedding to generate FiLM parameters to adapt the feature extractor.
        :param task_embedding: (torch.Tensor or None) Embedding of a whole task's context set.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (dict) Parameters of all FiLM layers. Empty dict if adapt_features=False.
        """
        film_dict = self.film_generator(task_embedding)

        if ops_counter:
            ops_counter.compute_macs(self.film_generator, task_embedding)

        return film_dict

    def predict(self, target_clips):
        """
        Function that processes target clips in batches to get logits over object classes for each clip.
        :param target_clips: (torch.Tensor) Target clips, each composed of self.clip_length contiguous frames.
        :return: (torch.Tensor) Logits over object classes for each clip in target_clips.
        """
        self._set_batch_norm_state()
        target_features = self._get_features_in_batches(target_clips, self.film_dict)
        target_features = self._pool_features(target_features)
        return self.classifier.predict(target_features)

    def predict_a_batch(self, target_clips):
        """
        Function that processes a single batch of target clips to get logits over object classes for each clip.
        :param target_clips: (torch.Tensor) Tensor of target clips, each composed of self.clip_length contiguous frames.
        :return: (torch.Tensor) Logits over object classes for each clip in target_clips.
        """
        self._set_batch_norm_state()
        target_features = self._get_features(target_clips, self.film_dict)
        target_features = self._pool_features(target_features)
        return self.classifier.predict(target_features) 
