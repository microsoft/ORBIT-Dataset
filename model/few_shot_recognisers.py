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

from data.utils import get_batch_indices
from features import extractors
from feature_adapters import FilmAdapter, NullAdapter
from model.poolers import MeanPooler
from model.set_encoder import SetEncoder, NullSetEncoder
from model.classifier_heads import LinearClassifier, VersaClassifier, PrototypicalClassifier, MahalanobisClassifier
from utils.optim import init_optimizer

class FewShotRecogniser(nn.Module):
    """
    Generic few-shot classification model.
    """
    def __init__(self, pretrained_extractor_path: str, feature_extractor: str,
        adapt_features: bool, classifier: str, clip_length: int, batch_size: int, learn_extractor: bool,
        feature_adaptation_method: str, logit_scale: float=1.0):
        """
        Creates instance of FewShotRecogniser.
        """
        super(FewShotRecogniser, self).__init__()

        self.adapt_features = adapt_features
        self.learn_extractor = learn_extractor
        self.clip_length = clip_length
        self.batch_size = batch_size
        self.feature_adaptation_method = feature_adaptation_method
        self.logit_scale = logit_scale

        # configure feature extractor
        extractor_fn = extractors[feature_extractor]
        self.feature_extractor = extractor_fn(
            pretrained=True if pretrained_extractor_path else False,
            pretrained_model_path=pretrained_extractor_path,
            with_film=self.adapt_features
        )
        if not self.learn_extractor:
            self._freeze_extractor()

        # configure feature adapter
        if self.adapt_features:
            if self.feature_adaptation_method == 'generate':
                self.set_encoder = SetEncoder()
                adaptation_layer = self.feature_extractor._get_adaptation_layer(generatable=True)
            else:
                self.set_encoder = NullSetEncoder()
                adaptation_layer = self.feature_extractor._get_adaptation_layer(generatable=False)
            self.feature_adapter = FilmAdapter(
                layer=adaptation_layer,
                adaptation_config = self.feature_extractor._get_adaptation_config(),
                task_dim=self.set_encoder.output_size
            )
        else:
            self.set_encoder = NullSetEncoder()
            self.feature_adapter = NullAdapter()

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

    def _get_features(self, clips, feature_adapter_params, ops_counter=None, context=False):
        """
        Function that passes clips through an adapted feature extractor to get adapted (and flattened) frame features.
        :param clips: (torch.Tensor) Tensor of clips, each composed of self.clip_length contiguous frames.
        :param feature_adapter_params: (list::dict::torch.Tensor or list::dict::list::torch.Tensor) Parameters of all FiLM layers.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param context: (bool) True if clips are from context videos, otherwise False.
        :return: (torch.Tensor) Adapted frame features flattened across all clips.
        """
        self._set_model_state(context)
        features = self.feature_extractor(clips, feature_adapter_params)

        if ops_counter:
            ops_counter.compute_macs(self.feature_extractor, clips, feature_adapter_params)

        return features

    def _get_features_in_batches(self, clips, feature_adapter_params, ops_counter=None, context=False):
        """
        Function that passes clips in batches through an adapted feature extractor to get adapted (and flattened) frame features.
        :param clips: (torch.Tensor) Clips, each composed of self.clip_length contiguous frames.
        :param feature_adapter_params: (list::dict::torch.Tensor or list::dict::list::torch.Tensor) Parameters of all FiLM layers.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param context: (bool) True if clips are from context videos, otherwise False.
        :return: (torch.Tensor) Adapted frame features flattened across all clips.
        """
        features = []
        self._set_model_state(context)

        num_clips = len(clips)
        num_batches = int(np.ceil(float(num_clips) / float(self.batch_size)))
        for batch in range(num_batches):
            batch_start_index, batch_end_index = get_batch_indices(batch, num_clips, self.batch_size)
            batch_clips = clips[batch_start_index:batch_end_index]
            if len(batch_clips.shape) == 5:
                batch_clips = batch_clips.flatten(end_dim=1)

            batch_clips = batch_clips.to(self.device, non_blocking=True)
            batch_features = self.feature_extractor(batch_clips, feature_adapter_params)

            if ops_counter:
                if self.adapt_features:
                    ops_counter.compute_macs(self.feature_extractor, batch_clips, feature_adapter_params)
                else:
                    ops_counter.compute_macs(self.feature_extractor, batch_clips)

            features.append(batch_features)

        return torch.cat(features, dim=0)

    def _get_feature_adapter_params(self, task_embedding, ops_counter=None):
        """
        Function that processes a task embedding to obtain parameters of the feature adapter.
        :param task_embedding: (torch.Tensor or None) Embedding of a whole task's context set.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: (list::dict::torch.Tensor or list::dict::list::torch.Tensor or None) Parameters of all FiLM layers.
        """
        feature_adapter_params = self.feature_adapter(task_embedding)

        if ops_counter:
            ops_counter.compute_macs(self.feature_adapter, task_embedding)

        return feature_adapter_params

    def _get_task_embedding(self, context_clips, ops_counter=None, reduction='mean'):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding.
        :param context_clips: (torch.Tensor) Tensor of context clips, each composed of self.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param reduction: (str) Method to aggregate clip encodings from self.set_encoder.
        :return: (torch.Tensor or None) Task embedding.
        """
        reps = self.set_encoder(context_clips)

        if ops_counter:
            ops_counter.compute_macs(self.set_encoder, context_clips)

        return self.set_encoder.aggregate(reps, reduction=reduction)

    def _get_task_embedding_in_batches(self, context_clips, ops_counter=None, reduction='mean'):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param reduction: (str) Method to aggregate clip encodings from self.set_encoder.
        :return: (torch.Tensor or None) Task embedding.
        """
        if not self.adapt_features:
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
                torch.cuda.synchronize()
                ops_counter.compute_macs(self.set_encoder, batch_clips)

            reps.append(batch_reps)

        return self.set_encoder.aggregate(reps, reduction=reduction)

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

    def _freeze_extractor(self):
        """
        Function that freezes all parameters in the feature extractor.
        :return: Nothing.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def _set_model_state(self, context=False):
        """
        Function that sets modules to appropriate train() or eval() states. Note, only modules that use batch norm (self.set_encoder, self.feature_extractor) and dropout (none) are affected.
        :param context: (bool) True if a context set is being processed, otherwise False.
        :return: Nothing.
        """
        self.set_encoder.train() # set encoder always in train mode (it processes context data)
        self.feature_extractor.eval()
        if self.learn_extractor and not self.test_mode:
            self.feature_extractor.train() # compute batch statistics in extractor if unfrozen and train mode

    def set_test_mode(self, test_mode):
        """
        Function that flags if model is being evaluated. Relevant for self._set_model_state().
        :param test_mode: (bool) True if model is being evaluated, otherwise True.
        :return: Nothing.
        """
        self.test_mode = test_mode

    def _reset(self):
        """
        Function that resets model's task-specific parameters after a task is processed.
        :return: Nothing.
        """
        self.feature_adapter_params = None
        self.classifier.reset()

class MultiStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot classification model that is personalised in multiple forward-backward steps (e.g. MAML, FineTuner).
    """
    def __init__(self, pretrained_extractor_path: str, feature_extractor: str,
        adapt_features: bool, classifier: str, clip_length: int, batch_size: int, learn_extractor: bool,
        feature_adaptation_method: str, num_grad_steps: int, logit_scale: float=1.0):
        """
        Creates instance of MultiStepFewShotRecogniser.
        """
        FewShotRecogniser.__init__(self, pretrained_extractor_path, feature_extractor,
            adapt_features, classifier, clip_length, batch_size, learn_extractor,feature_adaptation_method, logit_scale)

        self.num_grad_steps = num_grad_steps
    
    def personalise(self, context_clips, context_labels, learning_args, ops_counter=None):
        """
        Function that learns a new task by taking a fixed number of gradient steps on the task's full context set. For each task, a new linear classification layer is added (and FiLM layers if self.adapt_features == True).
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :param learning_args: (dict) Hyperparameters for personalisation.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        num_grad_steps = learning_args.pop('num_grad_steps')
        learning_rate = learning_args.pop('learning_rate')
        optimizer = learning_args.pop('optimizer')
        loss_fn = learning_args.pop('loss_fn')
        extractor_lr_scale = learning_args.pop('extractor_lr_scale')
        optimizer_kwargs = Namespace(**learning_args)

        num_classes = len(torch.unique(context_labels))
        self.init_classifier(num_classes)
        self.init_feature_adapter()
        personalize_optimizer = init_optimizer(self, learning_rate, optimizer, optimizer_kwargs, extractor_lr_scale)

        batch_context_set_size = len(context_labels)
        num_batches = int(np.ceil(float(batch_context_set_size) / float(self.batch_size)))

        for _ in range(num_grad_steps):
            for batch in range(num_batches):
                batch_start_index, batch_end_index = get_batch_indices(batch, batch_context_set_size, self.batch_size)
                batch_context_clips = context_clips[batch_start_index:batch_end_index].to(self.device)
                batch_context_labels = context_labels[batch_start_index:batch_end_index].to(self.device)
                batch_len = len(context_labels[batch_start_index:batch_end_index])

                feature_adapter_params = self._get_feature_adapter_params(None, ops_counter)
                batch_context_features = self._get_features(batch_context_clips, feature_adapter_params, ops_counter, context=True)
                batch_context_features = self._pool_features(batch_context_features, ops_counter)
                batch_context_logits = self.classifier.predict(batch_context_features, ops_counter)
                loss = loss_fn(batch_context_logits, batch_context_labels)
                loss += 0.001 * self.feature_adapter.regularization_term()
                loss *= batch_len/batch_context_set_size
                loss.backward()

            personalize_optimizer.step()
            personalize_optimizer.zero_grad()
    
    def predict(self, clips, ops_counter=None, context=False):
        """
        Function that processes target clips in batches to get logits over object classes for each clip.
        :param clips: (torch.Tensor) Clips, each composed of self.clip_length contiguous frames.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :param context: (bool) True if a context set is being processed, otherwise False.
        :return: (torch.Tensor) Logits over object classes for each clip in clips.
        """
        task_embedding = None # multi-step methods do not use set encoder
        feature_adapter_params = self._get_feature_adapter_params(task_embedding, ops_counter)
        features = self._get_features_in_batches(clips, feature_adapter_params, ops_counter, context=context)
        features = self._pool_features(features, ops_counter)
        return self.classifier.predict(features, ops_counter)

    def personalise_with_lite(self, context_clips, context_labels):
        NotImplementedError
    
    def init_classifier(self, num_classes:int):
        """
        Function that initialises classifier's parameters.
        :return: Nothing.
        """
        self.classifier.init(num_classes)
        self.classifier.to(self.device)

    def init_feature_adapter(self):
        """
        Function that initialises learnable FiLM layers
        :return: Nothing.
        """
        self.feature_adapter._init_layers()
        self.feature_adapter.to(self.device)

class SingleStepFewShotRecogniser(FewShotRecogniser):
    """
    Few-shot classification model that is personalised in a single forward step (e.g. CNAPs, ProtoNets).
    """
    def __init__(self, pretrained_extractor_path: str, feature_extractor: str,
        adapt_features: bool, classifier: str, clip_length: int, batch_size: int, learn_extractor: bool,
        feature_adaptation_method: str, num_lite_samples: int, logit_scale: float=1.0):
        """
        Creates instance of SingleStepFewShotRecogniser.
        """
        FewShotRecogniser.__init__(self, pretrained_extractor_path, feature_extractor,
            adapt_features, classifier, clip_length, batch_size, learn_extractor,feature_adaptation_method, logit_scale)
        self.num_lite_samples = num_lite_samples

    def personalise(self, context_clips, context_labels, ops_counter=None):
        """
        Function that learns a new task by performing a forward pass of the task's context set.
        :param context_clips: (torch.Tensor) Context clips each composed of self.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        task_embedding = self._get_task_embedding_in_batches(context_clips, ops_counter)
        self.feature_adapter_params = self._get_feature_adapter_params(task_embedding, ops_counter)
        context_features = self._get_features_in_batches(context_clips, self.feature_adapter_params, ops_counter, context=True)
        self.context_features = self._pool_features(context_features, ops_counter)
        self.classifier.configure(self.context_features, context_labels, ops_counter)

    def personalise_with_lite(self, context_clips, context_labels):
        """
        Function that learns a new task by performning a forward pass of the task's context set with LITE. Namely a random subset of the context set (self.num_lite_samples) is processed with back-propagation enabled, while the remainder is processed with back-propagation disabled.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param context_labels: (torch.Tensor) Video-level labels for each context clip.
        :return: Nothing.
        """
        shuffled_idxs = np.random.permutation(len(context_clips))
        H = self.num_lite_samples
        task_embedding = self._get_task_embedding_with_lite(context_clips[shuffled_idxs][:H], shuffled_idxs)
        self.feature_adapter_params = self._get_feature_adapter_params(task_embedding)
        self.context_features = self._get_pooled_features_with_lite(context_clips[shuffled_idxs][:H], shuffled_idxs)
        self.classifier.configure(self.context_features, context_labels[shuffled_idxs])

    def _cache_context_outputs(self, context_clips):
        """
        Function that performs a forward pass with a task's entire context set with back-propagation disabled and caches the individual 1) encodings from the set encoder and 2) adapted features from the adapted feature extractor, for each clip.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :return: Nothing.
        """
        with torch.set_grad_enabled(False):
            # cache encoding for each clip from self.set_encoder
            self.cached_set_encoder_reps = self._get_task_embedding_in_batches(context_clips, reduction='none')

            # get feature adapter parameters
            task_embedding = self.set_encoder.mean_pool(self.cached_set_encoder_reps)
            feature_adapter_params = self._get_feature_adapter_params(task_embedding)

            # cache adapted features for each clip
            context_features = self._get_features_in_batches(context_clips, feature_adapter_params, context=True)
            self.cached_context_features = self._pool_features(context_features)

    def _get_task_embedding_with_lite(self, context_clips, idxs):
        """
        Function that passes all of a task's context set through the set encoder to get a task embedding with LITE.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation enabled.
        :return: (torch.Tensor or None) Task embedding.
        """
        if isinstance(self.set_encoder, NullSetEncoder):
            return None
        H = self.num_lite_samples
        task_embedding_with_grads = self._get_task_embedding_in_batches(context_clips, reduction='none')
        task_embedding_without_grads = self.cached_set_encoder_reps[idxs][H:]
        return torch.cat((task_embedding_with_grads, task_embedding_without_grads)).mean(dim=0)

    def _get_pooled_features_with_lite(self, context_clips, idxs):
        """
        Function that gets adapted clip features for a task's context set with LITE.
        :param context_clips: (torch.Tensor) Context clips, each composed of self.clip_length contiguous frames.
        :param idxs: (torch.Tensor) Indicies of elements in context_clips to process with back-propagation enabled.
        :return: (torch.Tensor) Adapted frame features pooled per clip i.e. as (num_clips) x (feat_dim).
        """
        H = self.num_lite_samples
        context_features_with_grads = self._get_features_in_batches(context_clips, self.feature_adapter_params, context=True)
        context_features_with_grads = self._pool_features(context_features_with_grads)
        context_features_without_grads = self.cached_context_features[idxs][H:]
        return torch.cat((context_features_with_grads, context_features_without_grads))

    def predict(self, target_clips):
        """
        Function that processes target clips in batches to get logits over object classes for each clip.
        :param target_clips: (torch.Tensor) Target clips, each composed of self.clip_length contiguous frames.
        :return: (torch.Tensor) Logits over object classes for each clip in target_clips.
        """
        target_features = self._get_features_in_batches(target_clips, self.feature_adapter_params)
        target_features = self._pool_features(target_features)
        return self.classifier.predict(target_features)

    def predict_a_batch(self, target_clips):
        """
        Function that processes a single batch of target clips to get logits over object classes for each clip.
        :param target_clips: (torch.Tensor) Tensor of target clips, each composed of self.clip_length contiguous frames.
        :return: (torch.Tensor) Logits over object classes for each clip in target_clips.
        """
        target_features = self._get_features(target_clips, self.feature_adapter_params)
        target_features = self._pool_features(target_features)
        return self.classifier.predict(target_features)
