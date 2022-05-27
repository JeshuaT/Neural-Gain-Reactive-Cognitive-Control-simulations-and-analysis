import matplotlib.pyplot as plt
import numpy as np
import psyneulink as pnl
import pandas as pd

"""
This model simulates performance on the Stroop task over a range of gain values. The model is based on the Stroop GRAIN
model by Cohen & Huston (1994) (https://is.gd/hQaV5v) and further developed  by Jeshua Tromp to incorporate gain and 
thereby investigate the relationship between gain and cognitive control. The findings are now published in Tromp et al 
(2022) (https://rdcu.be/cOk1l).
"""

# CHOOSE WHICH SCRIPT TO EXECUTE ---------------------------------------------------------------------------------------
'''
Options to execute: 
"single"                -> single model execution 
"gain_grid"             -> model executions over a range of gain values (figure 3)
'''

run_sim = 'single'  # single or gain_grid

# DEFINE VARIABLES -----------------------------------------------------------------------------------------------------
# Gain parameters
gain = 1.0  # Gain parameter, in case of single simulation
gain_grid = np.around(np.arange(0.93, 1.07, 0.01), decimals=3)  # Gain grid, in case of multiple gain simulations

# Model parameters
rate = 0.01  # Integration constant
inhibition = -2.5  # Inhibition between units within the word, color, task and response layers
bias = -4  # Bias within the logistic function to color feature layer and word feature layer
threshold = 0.60  # # Threshold of activation accumulation in the response layer
settle = 100  # Number of trials for initialization period

# Transform values from cycles to RT
slope = 0.19
intercept = 589

# Container variables to store the results
if run_sim == 'single':
    pass
else:
    n_cong = []
    n_neutral = []
    n_incong = []

# MODEL BUILDING | NODES -----------------------------------------------------------------------------------------------
# Create 3 input layers: color, word & task
colors_input_layer = pnl.TransferMechanism(
    size=3,  # Define unit size
    function=pnl.Linear,
    name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(
    size=3,  # Define unit size
    function=pnl.Linear,
    name='WORDS_INPUT')

task_input_layer = pnl.TransferMechanism(
    size=2,  # Define unit size
    function=pnl.Linear,
    name='TASK_INPUT')

# Create color feature layer
colors_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,  # Define unit size
    function=pnl.Logistic(bias=bias),  # Activation function
    hetero=inhibition,  # Inhibition among units within a layer
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=rate,  # smoothing factor ==  integration rate
    name='COLORS HIDDEN')

# Create word feature layer
words_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,  # Define unit size
    function=pnl.Logistic(bias=bias),  # Activation function
    hetero=inhibition,  # Inhibition among units within a layer
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=rate,  # smoothing factor ==  integration rate
    name='WORDS HIDDEN')

# Create task-demand layer
task_layer = pnl.RecurrentTransferMechanism(
    size=2,  # Define unit size,
    function=pnl.Logistic(),  # Activation function
    hetero=inhibition,  # Inhibition among units within a layer
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=rate,  # smoothing factor ==  integration rate
    name='TASK')

#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(
    size=2,  # Define unit size
    function=pnl.Logistic(),  # Activation function
    hetero=inhibition,  # Inhibition among units within a layer
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=rate,  # smoothing factor ==  integration rate
    name='RESPONSE')

# MODEL BUILDING | WEIGHT LINES ----------------------------------------------------------------------------------------
# Input to task projection
task_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]))

# Color and word input start at 0 for initialization period
color_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))

word_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]))

# Color to task projection
color_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [4.0, 0.0],
        [4.0, 0.0],
        [4.0, 0.0]
    ]))

# Word to task projection
word_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 4.0],
        [0.0, 4.0],
        [0.0, 4.0]
    ]))

# Task to color projection
task_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0]
    ]))

# Task to word projection
task_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0]
    ]))

# Color to response projection
color_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.5, 0.0],
        [0.0, 1.5],
        [0.0, 0.0]
    ]))

# Word to response projection
word_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.5, 0.0],
        [0.0, 2.5],
        [0.0, 0.0]
    ]))

# Response to color projection, Cohen model has bidirectional response connections
response_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0]
    ]))

# Response to word projection, Cohen model has bidirectional response connections
response_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0]
    ]))

# MODEL BUILDING - PATHWAYS  -------------------------------------------------------------------------------------------
color_response_process_1 = pnl.Pathway(
    pathway=[
        colors_input_layer,
        color_input_weights,
        colors_hidden_layer,
        color_response_weights,
        response_layer,
    ],
    name='COLORS_RESPONSE_PROCESS_1')

color_response_process_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_color_weights,
        colors_hidden_layer
    ],
    name='COLORS_RESPONSE_PROCESS_2')

word_response_process_1 = pnl.Pathway(
    pathway=[
        words_input_layer,
        word_input_weights,
        words_hidden_layer,
        word_response_weights,
        response_layer
    ],
    name='WORDS_RESPONSE_PROCESS_1')

word_response_process_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_word_weights,
        words_hidden_layer
    ],
    name='WORDS_RESPONSE_PROCESS_2')

task_color_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_input_weights,
        task_layer,
        task_color_weights,
        colors_hidden_layer])

task_color_response_process_2 = pnl.Pathway(
    pathway=[
        colors_hidden_layer,
        color_task_weights,
        task_layer])

task_word_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_layer,
        task_word_weights,
        words_hidden_layer])

task_word_response_process_2 = pnl.Pathway(
    pathway=[
        words_hidden_layer,
        word_task_weights,
        task_layer])

# MODEL BUILDING | CREATE COMPOSITION ----------------------------------------------------------------------------------
Bidirectional_Stroop = pnl.Composition(
    pathways=[
        color_response_process_1,
        word_response_process_1,
        task_color_response_process_1,
        task_word_response_process_1,
        color_response_process_2,
        word_response_process_2,
        task_color_response_process_2,
        task_word_response_process_2
    ],
    reinitialize_mechanisms_when=pnl.Never(),
    name='Bidirectional Stroop Model')


# CREATE THRESHOLD FUNCTION --------------------------------------------------------------------------------------------
def pass_threshold(response_layer, thresh):
    results1 = response_layer.get_output_values(Bidirectional_Stroop)[0][0]  # Red response
    results2 = response_layer.get_output_values(Bidirectional_Stroop)[0][1]  # Green response
    # Return True if one of two response units passes threshold value
    if results1 >= thresh or results2 >= thresh:
        return True
    return False


# Stop trial if threshold is reached in response layer (pass_threshold function returns True)
terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, threshold)}

# CREATE LOGS ----------------------------------------------------------------------------------------------------------
task_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
response_layer.set_log_conditions('value')


# CREATE TEST TRIAL FUNCTION -------------------------------------------------------------------------------------------
# a BLUE word input is [1,0,0] to words_input_layer and GREEN word is [0,1,0] and neutral is [0,0,1]
# a blue color input is [1,0,0] to colors_input_layer and green color is [0,1,0] and neutral is [0,0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]
def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):
    trialdict = {
        colors_input_layer: [red_color, green_color, neutral_color],
        words_input_layer: [red_word, green_word, neutral_word],
        task_input_layer: [CN, WR]
    }
    return trialdict


# TRIAL FUNCTION -------------------------------------------------------------------------------------------------------
def run_trial(type_input, init_input, mod_gain):
    """
    Function to execute a trial of the model
    :param type_input: dict for congruent, neutral or incongruent trial
    :param init_input: dict for initialization input
    :param mod_gain: gain value
    :return: n cycles before reaching response threshold
    """
    # Set gain value
    print("simulating")
    print(mod_gain)
    task_layer.function_parameters.scale.set(mod_gain, context=Bidirectional_Stroop)
    colors_hidden_layer.function_parameters.scale.set(mod_gain, context=Bidirectional_Stroop)
    words_hidden_layer.function_parameters.scale.set(mod_gain, context=Bidirectional_Stroop)
    response_layer.function_parameters.scale.set(mod_gain, context=Bidirectional_Stroop)

    # Change color and word weights to 0 for the initialization run
    response_color_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]]),
        Bidirectional_Stroop)

    response_word_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]]),
        Bidirectional_Stroop)

    color_response_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]]),
        Bidirectional_Stroop)

    word_response_weights.parameters.matrix.set(
        np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]]),
        Bidirectional_Stroop)

    # Execute initialization run
    Bidirectional_Stroop.run(inputs=init_input, num_trials=settle)

    # Change color and word weights to back for trial run
    response_color_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]]),
        Bidirectional_Stroop)

    response_word_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0]]),
        Bidirectional_Stroop)

    color_response_weights.parameters.matrix.set(
        np.array([
            [1.5, 0.0],
            [0.0, 1.5],
            [0.0, 0.0]]),
        Bidirectional_Stroop)

    word_response_weights.parameters.matrix.set(
        np.array([
            [2.5, 0.0],
            [0.0, 2.5],
            [0.0, 0.0]]),
        Bidirectional_Stroop)

    # Execute model run and stop when response threshold has been reached (terminate trial value)
    Bidirectional_Stroop.run(inputs=type_input, termination_processing=terminate_trial)

    # Store values
    B_S = Bidirectional_Stroop.name
    r = response_layer.log.nparray_dictionary('value')  # Log response output from special logistic function
    rr = r[B_S]['value']
    n_r = rr.shape[0]
    rrr = rr.reshape(n_r, 2)
    n_type = rrr.shape[0]

    # Rename variables and store results
    if type_input == congruent_input:
        # Make variables global so its accessible outside the definition
        global n_cong
        if run_sim == 'single':
            n_cong = n_type
        else:
            n_cong.append(n_type)
    elif type_input == neutral_input:
        # Make variables global so its accessible outside the definition
        global n_neutral
        if run_sim == 'single':
            n_neutral = n_type
        else:
            n_neutral.append(n_type)
    elif type_input == incongruent_input:
        # Make variables global so its accessible outside the definition
        global n_incong
        if run_sim == 'single':
            n_incong = n_type
        else:
            n_incong.append(n_type)

    # CLEAR LOG & RESET ------------------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    colors_hidden_layer.log.clear_entries()
    words_hidden_layer.log.clear_entries()
    task_layer.log.clear_entries()
    colors_hidden_layer.reset([[0, 0, 0]])
    words_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0]])


# RUN TRIALS -----------------------------------------------------------------------------------------------------------
if run_sim == 'single':
    initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)
    congruent_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0)  # specify congruent trial input
    neutral_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0)  # create neutral trial stimuli input
    incongruent_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0)  # create incongruent trial stimuli input

    run_trial(neutral_input, initialize_input, gain)
    run_trial(congruent_input, initialize_input, gain)
    run_trial(incongruent_input, initialize_input, gain)
elif run_sim == 'gain_grid':
    for gain in gain_grid:
        initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)
        congruent_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0)  # specify congruent trial input
        neutral_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0)  # create neutral trial stimuli input
        incongruent_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0)  # create incongruent trial stimuli input

        run_trial(congruent_input, initialize_input, gain)
        run_trial(neutral_input, initialize_input, gain)
        run_trial(incongruent_input, initialize_input, gain)

# RESULTS --------------------------------------------------------------------------------------------------------------
"""
Per script results are shown
"single" --> Congruent, incongruent and neutral RT are shown, as well as interference and facilitation values
"gain_grid" --> A visualization of gain (x-axis) versus RT (y-axis) for congruent, neutral and incongruent trials.
                On top a pandas data-frame is made for further statistical analysis
"""
if run_sim == 'single':
    # Transfer cycles to seconds
    n_cong_RT = int((n_cong) * slope + intercept)
    n_neutral_RT = int((n_neutral) * slope + intercept)
    n_incong_RT = int((n_incong) * slope + intercept)

    interference = n_incong_RT - n_neutral_RT
    facilitation = n_neutral_RT - n_cong_RT

    print("Congruency RT: " + str(n_cong_RT),
          "\nNeutral RT: " + str(n_neutral_RT),
          "\nIncongruency RT: " + str(n_incong_RT),
          "\nInterference: " + str(interference),
          "\nFacilitation: " + str(facilitation))
elif run_sim == 'gain_grid':
    # Transfer cycles to RT in seconds
    n_cong_RT = [((i - settle) * slope + intercept) / 1000 for i in n_cong]
    n_neutral_RT = [((i - settle) * slope + intercept) / 1000 for i in n_neutral]
    n_incong_RT = [((i - settle) * slope + intercept) / 1000 for i in n_incong]

    # Making a pandas data-frame for further analysis
    add_0 = np.full(shape=np.size(n_cong), fill_value=0, dtype=int)
    add_1 = np.full(shape=np.size(n_cong), fill_value=1, dtype=int)
    congruent_df = pd.DataFrame({'RT': n_cong_RT, 'congruent': add_1, 'incongruent': add_0, 'gain': gain_grid})
    incongruent_df = pd.DataFrame({'RT': n_incong_RT, 'congruent': add_0, 'incongruent': add_1, 'gain': gain_grid})
    neutral_df = pd.DataFrame({'RT': n_neutral_RT, 'congruent': add_0, 'incongruent': add_0, 'gain': gain_grid})
    to_merge = [congruent_df, neutral_df, incongruent_df]
    sim_RT_df = pd.concat(to_merge, ignore_index=True)
    sim_RT_df['logRT'] = np.log(sim_RT_df['RT'])
    df = sim_RT_df
    conditions = [
        (df['congruent'] == 1) & (df['incongruent'] == 0),
        (df['congruent'] == 0) & (df['incongruent'] == 1),
        (df['congruent'] == 0) & (df['incongruent'] == 0)]
    choices = ['congruent', 'incongruent', 'neutral']
    sim_RT_df['condition'] = np.select(conditions, choices)  # Final dataframe for further plotting/analysis

    # START HERE IF YOU IMPORT THE DATAFRAME ---------------------------------------------------------------------------
    # FONT parameters
    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 23}

    plt.rc('font', **font)

    # Make subsets of dataframe for calculation interference/facilitation/stroop effects
    congruent = sim_RT_df[sim_RT_df['condition'] == "congruent"]
    congruent.reset_index(inplace=True)

    neutral = sim_RT_df[sim_RT_df['condition'] == "neutral"]
    neutral.reset_index(inplace=True)

    incongruent = sim_RT_df[sim_RT_df['condition'] == "incongruent"]
    incongruent.reset_index(inplace=True)

    # Compute interference, facilitation and Stroop effect for each condition
    interference = incongruent['logRT'] - neutral['logRT']
    facilitation = neutral['logRT'] - congruent['logRT']
    stroop_effect = incongruent['logRT'] - congruent['logRT']

    # Make gain-RT plot
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)  # Remove top edge
    ax.spines['right'].set_visible(False)  # Remove right edge
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_zorder(0)
    plt.plot(gain_grid, congruent['logRT'], 'g-', linewidth=6, label='Congruent')  # label='Congruent'
    plt.plot(gain_grid, neutral['logRT'], 'k-', linewidth=6, label='Neutral')  # label='Neutral'
    plt.plot(gain_grid, incongruent['logRT'], 'r-', linewidth=6, label='Incongruent')  # label='Incongruent'
    plt.xlabel('gain')
    plt.xticks(ticks=[0.95, 1.00, 1.05])
    plt.xlim(0.93, 1.07)
    plt.ylabel('log RT(s)')
    plt.ylim(-0.6, 0.0)
    plt.yticks(ticks=[-0.6, -0.3, 0.0])
    # plt.title('Simulated RT with varying gain')
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()  # So that the labels don't fall out of the frame
    plt.show()

    # Make gain facilitation/interference/stroop effect plot
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)  # Remove top edge
    ax.spines['right'].set_visible(False)  # Remove right edge
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_zorder(0)
    plt.plot(gain_grid, interference, 'b-', linewidth=4,
             label='Interference')  # label='Interference effect (INCON - NEU)')
    plt.plot(gain_grid, facilitation, 'm--', linewidth=4,
             label='Facilitation')  # label='Facilitation effect (NEU - CON)')
    # plt.plot(gain_grid, stroop_effect, 'go', label='_nolegend_')  # label='Stroop effect (INCON - CON)')
    plt.xlabel('gain')
    plt.xticks(ticks=[0.95, 1.00, 1.05])
    plt.xlim(0.93, 1.07)
    plt.ylabel('Δ log RT(s)')
    plt.ylim(-0.0, 0.20)
    plt.yticks(ticks=[0.0, 0.1, 0.2])
    # plt.title('Interference, facilitation and stroop effect with varying gain')
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()  # So that the labels don't fall out of the frame
    plt.show()

    # Save files
    filename = "df_cohenhuston1994_gaingrid"
    sim_RT_df.to_csv(
        "{0}".format(filename),
        index=False)
