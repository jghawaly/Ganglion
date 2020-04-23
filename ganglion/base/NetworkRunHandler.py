from NeuralNetwork import NeuralNetwork
from NeuralGroup import *
from timekeeper import *
from units import *
from utils import poisson_train, add_noise
import os
import numpy as np


class NetworkRunHandler:
    # output encoding mechanisms
    FIRST_OUTPUT_ENC = 0
    MULTI_FIRST_OUTPUT_ENC = 1
    STEADY_STATE_OUTPUT_ENC = 2
    TIMED_ENC = 3
    COUNT_ENC = 4
    
    def __init__(self, nn: NeuralNetwork, 
                       training_data,
                       training_labels,
                       network_labels,
                       run_order,
                       output_encoding=0,
                       enable_additive_frequency=True,
                       enable_subtractive_frequency=True,
                       base_input_frequency=5.0,
                       additive_frequency_boost=5.0,
                       additive_frequency_wait=50.0,
                       enable_noise_sampling=False,
                       noise_sampling_interval=3.0,
                       noise_probability=0.1,
                       normalize_on_weight_change=True,
                       normalize_on_start=True,
                       episodes=None,
                       save_at_end=False,
                       save_dir=None,
                       reset_on_process=True,
                       rewards=(-1.0, 1.0),
                       data_pre_processor=None,
                       training=True,
                       exposure_hits=10,
                       output_is_ftm=False,
                       ftm_supervised=False,
                       allow_multi_wrong=False, 
                       encoding_time=50 * msec,
                       decision_encoding_time=10 * msec,
                       encoding_count=20,
                       no_label=False,
                       move_on_time=300*msec, 
                       verbose=False,
                       decay_weights=False,
                       store_predictions=False,
                       ros=False):
        # Network Properties
        self.nn = nn
        self.run_order = run_order
        self.tki = nn.tki
        self.dt = self.tki.dt()
        self.network_labels = network_labels
        self.output_encoding = output_encoding
        self.exposure_hits = exposure_hits  # only used in MULTI_FIRST_OUTPUT_ENC encoding mode
        self.hit_count = 0  # only used in MULTI_FIRST_OUTPUT_ENC encoding mode
        self.no_label=no_label
        self.decay_weights = decay_weights

        # Network Properties When Output is FTM
        self.output_is_ftm = output_is_ftm
        self.ftm_supervised = ftm_supervised

        # For TIMED_ENC only
        self.initial_encoding_time = encoding_time  # amount of time to integrate spikes for the stabilization stage
        self.tsi = 0.0  # amount of time that we have been integrating output spikes for in the stabilization stage
        self.decision_encoding_time = decision_encoding_time  # amount of time to integrate spikes for the decision phase

        # For COUNT_ENC only
        self.encoding_count = encoding_count  # number of output spikes per input
        self.time_in_ep = 0
        self.move_on_time = move_on_time  # if the requested number of spikes is not registered in this amount of time, move on to the next example

        # Data pre-processors
        self.data_pre_processor = data_pre_processor

        # Training data
        self.training = training
        self.data_counter = -1
        self.training_data = training_data
        self.training_labels = training_labels
        self.on_next_input()  # get the first input and label
        
        # Standard 
        self.base_input_frequency = base_input_frequency
        self.reset_on_process = reset_on_process
        self.rewards = rewards
        self.cummulative_spikes = np.zeros(self.nn.neural_groups[-1].shape)
        self.ros = ros
        if episodes is None:
            self.episodes = training_data.shape[0]
        else:
            self.episodes = episodes
        self.current_episode = 0
        self.time_since_eval = 0
        self.output_oversaturated = False  # sets to True on multiple, simultaneous spike
        self.allow_mw = allow_multi_wrong  # True to allow the network to make multiple wrong decisions at once
        self.verbose = verbose
        self.store_predictions = store_predictions
        self.y_true = []
        self.y_pred = []

        # Additive frequency 
        self.enable_additive_frequency = enable_additive_frequency
        self.additive_frequency_boost = additive_frequency_boost
        self.additive_frequency_wait = additive_frequency_wait
        self.ft_add = 0.0
        self.time_since_process = 0.0

        # Subtractive frequency
        self.enable_subtractive_frequency = enable_subtractive_frequency

        # Noise sampling 
        self.enable_noise_sampling = enable_noise_sampling
        self.noise_sampling_interval = noise_sampling_interval
        self.noise_probability = noise_probability
        self.last_noise_sampling_tick = 0

        # Weight normalization
        self.normalize_on_weight_change = normalize_on_weight_change
        self.normalize_on_start = normalize_on_start

        # network weight saving
        self.save_at_end = save_at_end
        self.save_dir = save_dir

        # run on_start
        self.on_start()

        # last_executed_tick
        self.tick = 0

        # this gets changed and stored after each episode
        self.metrics = None

    def on_start(self):
        """
        This method is called at the end of the initialization process
        """
        if self.normalize_on_start and self.training:
            self.nn.normalize_weights()
    
    def on_end(self):
        """
        This method is called after the completion of all training episodes
        """
        if self.save_at_end:
            # save every weight in the network
            for s in self.nn.synapses:
                input_group_name = s.pre_n.name 
                output_group_name = s.post_n.name 
                path = os.path.join(self.save_dir, "%s_%s.npy" % (input_group_name, output_group_name))
                self.nn.save_w(path, input_group_name, output_group_name)
    
    def on_noise_sample(self):
        """
        Called when noise sampling interval is triggered
        """
        self.current_input = self.current_input_unprocessed.copy()
        self.last_noise_sampling_tick = self.tick
        self.current_input = add_noise(self.current_input, p=self.noise_probability)
    
    def on_additive_frequency(self):
        """
        Called when a request has been made to increase the input frequency
        """
        self.time_since_eval += self.dt 
        if self.time_since_eval >= self.additive_frequency_wait * msec:
            self.ft_add += self.additive_frequency_boost
            self.time_since_eval = 0
    
    def on_subtractive_frequency(self):
        self.ft_add -= self.additive_frequency_boost

    def on_next_input(self):
        """
        Called when a new input is to be generated
        """
        self.data_counter += 1
        self.time_in_ep = 0

        # make sure we don't exceed the bounds of the input data array
        if self.data_counter == self.training_data.shape[0]:
            self.data_counter = 0
        
        self.current_input = self.training_data[self.data_counter]
        self.current_label = self.training_labels[self.data_counter]

        # run pre-processing steps on data
        if self.data_pre_processor is not None:
            self.current_input = self.data_pre_processor(self.current_input)
        
        self.current_input_unprocessed = self.current_input.copy()

        if self.output_is_ftm:
            self.nn.neural_groups[-1].reset_ftm()
            self.nn.neural_groups[-1].ftm_mod(self.network_labels, self.current_label)

    def on_process_output(self):
        """
        Called when we need to process the output of the network
        """
        if not self.no_label:
            # action = self.cummulative_spikes.argmax()
            action = self.network_labels[np.argmax([self.cummulative_spikes[self.network_labels==v].sum() for v in np.unique(self.network_labels)])]

            # True for correct prediction, False otherwise
            correct = action == self.current_label
            
            # determine reward
            reward = self.rewards[1] if correct else self.rewards[0]
            if self.store_predictions:
                self.y_true.append(self.current_label)
                self.y_pred.append(action)

            if self.verbose:
                print("Target: %s" % str(self.current_label))
                print("Action: %s" % str(action))
                print("Reward: %s" % str(reward))
            
            # puff some dopamine into the network
            if self.training:
                self.nn.dopamine_puff(reward, actions=action)

            # calculate accuracy
            acc = 1.0 if correct else 0.0
            
            return reward, acc
        else:
            return 0.0, 0.0
    
    def on_post_process(self):
        # if requested, perform weight decay
        if self.decay_weights:
            self.nn.decay_weights()

        # if requested, normalize the weights
        if self.normalize_on_weight_change and self.training:
            self.nn.normalize_weights()

        # get our next input
        self.on_next_input()
        
        # iterate episode
        self.current_episode += 1
        if self.verbose:    
            print("Curret Episode: %s" % str(self.current_episode))

        # rest cummulative spikes of output
        self.cummulative_spikes.fill(0)

    def step(self):
        """
        Performs the necessary step to run the network. Returns False when the episode is over, and True otherwise
        """
        # iterate tick
        self.tick = self.tki.__next__()

        # check if we are doing noise sampling
        if self.enable_noise_sampling:
            # check if it is time to sample noise
            if (self.tick - self.last_noise_sampling_tick) * self.dt >= self.noise_sampling_interval * msec:
                self.on_noise_sample()
        
        # check first spike output
        if self.output_encoding == self.__class__.FIRST_OUTPUT_ENC:
            if self.cummulative_spikes.sum() > 0:
                if  np.count_nonzero(self.cummulative_spikes == self.cummulative_spikes.max()) == 1:
                    # process our output spikes
                    r, a = self.on_process_output()

                    # if requested, reset the network
                    if self.reset_on_process:
                        self.nn.reset()

                    # if requested, zero out the additive frequency since we got spikes
                    if self.enable_additive_frequency:
                        self.ft_add = 0
                    
                    # set metrics
                    self.metrics = (r, a, self.cummulative_spikes.sum())

                    # get our next input
                    self.on_post_process()

                    return False
                else:
                    self.nn.reset()
            else:
                # if requested, perform additive frequency
                if self.enable_additive_frequency:
                    self.on_additive_frequency()
        
        if self.output_encoding == self.__class__.MULTI_FIRST_OUTPUT_ENC:
            if self.cummulative_spikes.sum() > 0:
                c = np.count_nonzero(self.cummulative_spikes == self.cummulative_spikes.max())
                if c == 1:
                    # increase hit count
                    self.hit_count += 1

                    # process our output spikes
                    r, a = self.on_process_output()
                    
                    # if requested, zero out the additive frequency since we got spikes
                    if self.enable_additive_frequency:
                        self.ft_add = 0
                    
                    # set metrics
                    self.metrics = (r, a, self.cummulative_spikes.sum())

                    # clear out cummulative spike array
                    self.cummulative_spikes.fill(0)

                    if self.hit_count == self.exposure_hits:
                        self.nn.reset()
                        self.on_post_process()
                        self.hit_count = 0
                        return False
                    
                    # output not saturated
                    self.output_oversaturated = False
                elif c > 1:
                    # forget everything that just happened
                    self.nn.reset()
                    # clear out cummulative spike array
                    self.cummulative_spikes.fill(0)
                    # set saturation flag to True
                    self.output_oversaturated = True
                    if self.enable_subtractive_frequency:
                        self.on_subtractive_frequency()
            else:
                # if requested, perform additive frequency
                if self.enable_additive_frequency and not self.output_oversaturated:
                    self.on_additive_frequency()
                    # reset saturation
                    self.output_oversaturated = False

        if self.output_encoding == self.__class__.TIMED_ENC:
            # increase integration time
            self.tsi += self.tki.dt()

            # if we have been integrating long enough
            if self.tsi >= self.initial_encoding_time:
                if self.tsi >= self.initial_encoding_time + self.decision_encoding_time:
                    r, a = self.on_process_output()
                    # set metrics
                    self.metrics = (r, a, self.cummulative_spikes.sum())
                    if self.reset_on_process:
                        self.nn.reset()
                    self.on_post_process()
                    self.tsi = 0.0
                    return False
                else:
                    self.cummulative_spikes.fill(0)
        
        if self.output_encoding == self.__class__.COUNT_ENC:
            self.time_in_ep += self.tki.dt() 
            if self.time_in_ep > self.move_on_time:
                self.nn.reset()
                self.on_post_process()
                self.tsi = 0
                if self.enable_additive_frequency:
                    self.ft_add = 0
                if self.verbose:
                    print("...Max time per episode exceeded, moving on to next input...")
                return False
            spike_sum = self.cummulative_spikes.sum()
            if spike_sum > 0:
                
                # if we have been integrating long enough
                if spike_sum >= self.encoding_count:
                    r, a = self.on_process_output()
                    # set metrics
                    self.metrics = (r, a, self.cummulative_spikes.sum())
                    if self.reset_on_process:
                        self.nn.reset()
                    self.on_post_process()
                    self.tsi = 0.0

                    # if requested, zero out the additive frequency since we got spikes
                    if self.enable_additive_frequency:
                        self.ft_add = 0
                    return False
                else:
                    if self.ros and (self.nn.neural_groups[-1].spike_count.sum() > 0):
                        self.nn.reset()
            else:
                # if requested, perform additive frequency
                if self.enable_additive_frequency:
                    self.on_additive_frequency()

        # inject spikes into sensory layer
        self.nn.neural_groups[0].run(poisson_train(self.current_input, self.dt, (self.base_input_frequency + self.ft_add) / self.current_input.max()))

        # run all layers
        self.nn.run_order(self.run_order)

        # accumulate output spikes
        self.cummulative_spikes += self.nn.neural_groups[-1].spike_count

        return True
    
    def run_episode(self):
        """
        Loops over step() until the episode is over. Return True, metrics when it has finished running the episode AND we have not finished ALL episodes. When all
        episodes are over, it returns False, metrics
        """
        # check if we are done running
        if self.current_episode == self.episodes:
            self.on_end()
            return False, self.metrics

        # while we are running an episode, wait
        running = True
        while running:
            running = self.step()
        
        # when the episode is done, return the results
        return True, self.metrics
    
    def get_predictions(self):
        return self.y_true, self.y_pred
