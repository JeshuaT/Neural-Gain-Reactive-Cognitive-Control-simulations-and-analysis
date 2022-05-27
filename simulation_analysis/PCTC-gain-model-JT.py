import matplotlib.pyplot as plt
import numpy as np
import psyneulink as pnl
import pandas as pd

"""
This model simulates performance on the Stroop task over a single, or a range of gain values. The model is based on the 
PCTC model developed by Kalanthroff et al (2018) (https://psycnet.apa.org/record/2017-46559-001) and further developed 
by Jeshua Tromp to incorporate gain and thereby investigate the relationship between gain and cognitive control. The 
findings are now published in Tromp et al (2022) (https://rdcu.be/cOk1l).
"""

# CHOOSE WHICH SCRIPT TO EXECUTE ---------------------------------------------------------------------------------------
'''
Options to execute: 
"single"                -> single model execution (figure 2) 
"gain_grid"             -> model executions over a range of gain values (figure 1)
'''

run_sim = 'single'

# DEFINE VARIABLES -----------------------------------------------------------------------------------------------------
# Gain parameters
gain = 1.0  # Gain parameter, in case of single simulation
gain_grid = np.around(np.arange(0.90, 1.1, 0.01), decimals=3)  # Gain grid, in case of multiple gain simulations

# Model parameters
pc = 0.124  # Pro-active control (0.124)
inhibition_task = -1.97  # Inhibition between units within the task layer
inhibition = -1.3  # Inhibition between units within the word, color and response layers
Lambda = 0.03  # Integration constant (PsyNeuLink has Euler integration constant reversed (1-0.97))
bias = -0.3  # Bias input to color feature layer and word feature layer
threshold = 0.70  # Threshold of activation accumulation in the response layer
settle = 500  # Number of trials for initialization period

# Activation function parameters
log_bias = -1  # Bias within the logistic function
log_gain = 4  # Gain within the logistic function. This gain is a fixed property of the activation function,
# not a changeable gain parameter within our framework
scale = 1.0  # Standard multiplicative gain factor

# Transform values from cycles to RT
slope = 1.39
intercept = 211

# Container variables to store the results
if run_sim == 'single':
    pass
else:
    n_cong = []
    n_neutral = []
    n_incong = []


# MODEL FUNCTIONS ------------------------------------------------------------------------------------------------------
# This function constrains input to be at least 0
def my_special_Logistic(variable):
    """
    :param variable: log-transformed activation value
    :return: constrains input to be 0
    """
    maxi = variable - 0.0180
    value = np.fmax([0], maxi)
    output = np.fmin([1], value)
    return output


# This function computes the conflict inhibition to the response layer that arises if both task-units are activated.
# The conflict value is multiplied by 500, increasing the conflict by an amount that fully inhibits the response layer.
def my_conflict_function(variable):
    """
    :param variable: array of activation of both word-reading and color-naming nodes
    :return: value of conflict (is 0 when either one of the nodes does not have activity)
    """
    maxi = variable - 0.0180
    new = np.fmax([0], maxi)
    out = [new[0] * new[1] * 500]
    return out


# MODEL BUILDING | NODES -----------------------------------------------------------------------------------------------
# Create 4 input layers: color, word, task & bias
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
    name='PROACTIVE_CONTROL')

bias_input = pnl.TransferMechanism(
    size=3,  # Define unit size
    function=pnl.Linear,
    name='BIAS')

# Create color feature layer
color_feature_layer = pnl.RecurrentTransferMechanism(
    size=3,  # Define unit size
    function=pnl.Logistic(gain=log_gain, bias=log_bias, scale=scale),  # Activation function
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=Lambda,  # smoothing factor ==  integration rate
    hetero=inhibition,  # Inhibition among units within a layer
    output_ports=[{  # Create new OutputPort by applying the "my_special_Logistic" function
        pnl.NAME: 'SPECIAL_LOGISTIC',
        pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
        pnl.FUNCTION: my_special_Logistic}],
    name='COLOR_LAYER')

# Create word feature layer
word_feature_layer = pnl.RecurrentTransferMechanism(
    size=3,  # Define unit size
    function=pnl.Logistic(gain=log_gain, bias=log_bias, scale=scale),  # Activation function
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=Lambda,  # smoothing factor ==  integration rate
    hetero=inhibition,  # Inhibition among units within a layer
    output_ports=[{  # Create new OutputPort by applying the "my_special_Logistic" function
        pnl.NAME: 'SPECIAL_LOGISTIC',
        pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
        pnl.FUNCTION: my_special_Logistic}],
    name='WORD_LAYER')

# Create task-demand layer
# Note: added output port (CONFLICT) and different lateral inhibition weight than other layers
task_demand_layer = pnl.RecurrentTransferMechanism(
    size=2,  # Define unit size
    function=pnl.Logistic(gain=log_gain, bias=log_bias, scale=scale),  # Activation function
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=Lambda,  # smoothing factor ==  integration rate
    hetero=inhibition_task,  # Inhibition among units within a layer
    output_ports=[{  # Create new OutputPort by applying the "my_special_Logistic" function
        pnl.NAME: 'SPECIAL_LOGISTIC',
        pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
        pnl.FUNCTION: my_special_Logistic},
        {
            pnl.NAME: 'CONFLICT',  # If there is conflict this inhibits response
            pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
            pnl.FUNCTION: my_conflict_function}],
    name='TASK_LAYER')

# Create response-layer
response_layer = pnl.RecurrentTransferMechanism(
    size=2,  # Define unit size
    function=pnl.Logistic(gain=log_gain, bias=log_bias, scale=scale),  # Activation function
    integrator_mode=True,  # Set IntegratorFunction mode to True
    integration_rate=Lambda,  # smoothing factor ==  integration rate
    hetero=inhibition,  # Inhibition among units within a layer
    output_ports=[{  # Create new OutputPort by applying the "my_special_Logistic" function
        pnl.NAME: 'SPECIAL_LOGISTIC',
        pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
        pnl.FUNCTION: my_special_Logistic}],
    name='RESPONSE_LAYER')

# MODEL BUILDING | WEIGHT LINES ----------------------------------------------------------------------------------------
# Color and word input start at 0 for initialization period
color_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]))

word_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]))

# Color to task projection
color_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.0]
    ]))

# Word to task projection
word_task_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 2.0],
        [0.0, 2.0],
        [0.0, 2.0]
    ]))

# Task to color projection
task_color_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ]))

# Task to word projection
task_word_weights = pnl.MappingProjection(
    matrix=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ]))

# Color to response projection
color_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.0, 0.0],
        [0.0, 2.0],
        [0.0, 0.0]
    ]))

# Word to response projection
word_response_weights = pnl.MappingProjection(
    matrix=np.array([
        [2.5, 0.0],
        [0.0, 2.5],
        [0.0, 0.0]
    ]))

# Proactive control to task projection
task_input_weights = pnl.MappingProjection(
    matrix=np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ]))

# Send CONFLICT signal from the task demand layer to the response layer
task_conflict_to_response_weights = pnl.MappingProjection(
    matrix=np.array([[-1.0, -1.0]]),
    sender=task_demand_layer.output_ports[1],
    receiver=response_layer)

# MODEL BUILDING - PATHWAYS  -------------------------------------------------------------------------------------------
color_response_process = pnl.Pathway(pathway=[
    colors_input_layer,
    color_input_weights,
    color_feature_layer,
    color_response_weights,
    response_layer],
    name='COLORS_RESPONSE_PROCESS')

word_response_process = pnl.Pathway(
    pathway=[words_input_layer,
             word_input_weights,
             word_feature_layer,
             word_response_weights,
             response_layer],
    name='WORDS_RESPONSE_PROCESS')

task_color_process = pnl.Pathway(
    pathway=[task_input_layer,
             task_input_weights,
             task_demand_layer,
             task_color_weights,
             color_feature_layer,
             color_task_weights,
             task_demand_layer],
    name='TASK_COLOR_PROCESS')

task_word_process = pnl.Pathway(
    pathway=[task_input_layer,
             task_demand_layer,
             task_word_weights,
             word_feature_layer,
             word_task_weights,
             task_demand_layer],
    name='TASK_WORD_PROCESS')

bias_color_process = pnl.Pathway(
    pathway=[bias_input,
             color_feature_layer],
    name='BIAS_COLOR')

bias_word_process = pnl.Pathway(
    pathway=[bias_input,
             word_feature_layer],
    name='WORD_COLOR')

conflict_process = pnl.Pathway(
    pathway=[
        task_demand_layer,
        task_conflict_to_response_weights,
        response_layer],
    name='CONFLICT_PROCESS')

# MODEL BUILDING | CREATE COMPOSITION ----------------------------------------------------------------------------------
PCTC = pnl.Composition(
    pathways=[
        word_response_process,
        color_response_process,
        task_color_process,
        task_word_process,
        bias_word_process,
        bias_color_process,
        conflict_process],
    reinitialize_mechanisms_when=pnl.Never(),
    name='PCTC_MODEL')


# CREATE THRESHOLD FUNCTION --------------------------------------------------------------------------------------------
def pass_threshold(layer, thresh, context):
    results1 = layer.get_output_values(context)[0][0]  # Red response
    results2 = layer.get_output_values(context)[0][1]  # Green response
    # Return True if one of two response units passes threshold value
    if results1 >= thresh or results2 >= thresh:
        return True
    return False


# Stop trial if threshold is reached in response layer (pass_threshold function returns True)
terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.While(pass_threshold, response_layer, threshold)}

# CREATE LOGS ----------------------------------------------------------------------------------------------------------
color_feature_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output color feature layer
word_feature_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output word feature layer
response_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output response layer
task_demand_layer.set_log_conditions('SPECIAL_LOGISTIC')  # Log output task demand layer

task_demand_layer.set_log_conditions('CONFLICT')  # Log output of conflict inhibition


# CREATE TEST TRIAL FUNCTION -------------------------------------------------------------------------------------------
# a BLUE word input is [1,0,0] to words_input_layer, GREEN word is [0,1,0] and NEUTRAL is [0,0,1]
# a blue color input is [1,0,0] to colors_input_layer, green color is [0,1,0] and neutral is [0,0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]
def trial_dict(blue_color, green_color, neutral_color, blue_word, green_word, neutral_word, pc_cn, pc_wr, bias_trial):
    trialdict = {
        colors_input_layer: [blue_color, green_color, neutral_color],
        words_input_layer: [blue_word, green_word, neutral_word],
        task_input_layer: [pc_cn, pc_wr],
        bias_input: [bias_trial, bias_trial, bias_trial]
    }
    return trialdict


# TRIAL FUNCTION -------------------------------------------------------------------------------------------------------
def run_trial(type_input, init_input, mod_gain):
    """
    Function to execute a trial of the model
    :param type_input: dict for congruent, neutral or incongruent trial
    :param init_input: dict for initialization input (congruent trial)
    :param mod_gain: gain value
    :return: n cycles until reaching response threshold
    """
    print("simulating")
    # Set gain value
    task_demand_layer.function_parameters.scale.set(mod_gain, context=PCTC)
    response_layer.function_parameters.scale.set(mod_gain, context=PCTC)
    word_feature_layer.function_parameters.scale.set(mod_gain, context=PCTC)
    color_feature_layer.function_parameters.scale.set(mod_gain, context=PCTC)

    # Change color and word weights to 0 for the initialization run
    color_input_weights.parameters.matrix.set(
        np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]), PCTC)

    word_input_weights.parameters.matrix.set(
        np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]), PCTC)

    # Execute initialization run
    PCTC.run(inputs=init_input, num_trials=settle)

    # Set color and word weights back to 1 for model to process stimulus projections
    color_input_weights.parameters.matrix.set(
        np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]), PCTC)

    word_input_weights.parameters.matrix.set(
        np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]), PCTC)

    # Execute model run and stop when response threshold has been reached (terminate trial value)
    PCTC.run(inputs=type_input, termination_processing=terminate_trial)

    # Store values
    # Log TASK output from special logistic function
    t = task_demand_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
    tt = t[PCTC.name]['SPECIAL_LOGISTIC']
    n_type = tt.shape[0]
    ttt_type = tt.reshape(n_type, 2)

    # Compute Conflict for plotting
    conflict_type = ttt_type[settle:, 0] * ttt_type[settle:, 1] * 100

    # Log RESPONSE output from special logistic function
    r = response_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
    rr = r[PCTC.name]['SPECIAL_LOGISTIC']
    rrr_type = rr.reshape(n_type, 2)
    res = rrr_type[-1]

    # Log COLOR output from special logistic function
    c = color_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
    cc = c[PCTC.name]['SPECIAL_LOGISTIC']
    ccc_type1 = cc.reshape(1, n_type, 3)
    # Remove dimension from array
    ccc_type = np.squeeze(ccc_type1)

    # Log WORD output from special logistic function
    w = word_feature_layer.log.nparray_dictionary('SPECIAL_LOGISTIC')
    ww = w[PCTC.name]['SPECIAL_LOGISTIC']
    www_type1 = ww.reshape(1, n_type, 3)
    # Remove dimension from array
    www_type = np.squeeze(www_type1)

    # Rename variables and store results
    if type_input == congruent_input:
        # Make variables global so its accessible outside the definition
        global n_cong, ttt_cong, rrr_cong, ccc_cong, www_cong, conflict_cong
        if run_sim == 'single':
            n_cong = n_type
            ttt_cong = ttt_type
            rrr_cong = rrr_type
            ccc_cong = ccc_type
            www_cong = www_type
            conflict_cong = conflict_type
        else:
            n_cong.append(n_type)
    elif type_input == neutral_input:
        # Make variables global so its accessible outside the definition
        global n_neutral, ttt_neutral, rrr_neutral, ccc_neutral, www_neutral, conflict_neutral
        if run_sim == 'single':
            n_neutral = n_type
            ttt_neutral = ttt_type
            rrr_neutral = rrr_type
            ccc_neutral = ccc_type
            www_neutral = www_type
            conflict_neutral = conflict_type
        else:
            n_neutral.append(n_type)
    elif type_input == incongruent_input:
        # Make variables global so its accessible outside the definition
        global n_incong, ttt_incong, rrr_incong, ccc_incong, www_incong, conflict_incong
        if run_sim == 'single':
            n_incong = n_type
            ttt_incong = ttt_type
            rrr_incong = rrr_type
            ccc_incong = ccc_type
            www_incong = www_type
            conflict_incong = conflict_type
        else:
            n_incong.append(n_type)

    # CLEAR LOG & RESET ------------------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    color_feature_layer.log.clear_entries()
    word_feature_layer.log.clear_entries()
    task_demand_layer.log.clear_entries()

    color_feature_layer.reset([[0, 0, 0]], context=PCTC)
    word_feature_layer.reset([[0, 0, 0]], context=PCTC)
    response_layer.reset([[0, 0]], context=PCTC)
    task_demand_layer.reset([[0, 0]], context=PCTC)


# RUN TRIALS -----------------------------------------------------------------------------------------------------------
if run_sim == 'single':
    initialize_input = trial_dict(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pc, 0.0, bias)
    # RUN TRIALS -------------------------------------------------------------------------------------------------
    congruent_input = trial_dict(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, pc, 0.0, bias)  # Create cong trial input
    neutral_input = trial_dict(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, pc, 0.0, bias)  # Create neutral trial stimuli input
    incongruent_input = trial_dict(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, pc, 0.0, bias)  # Create incong trial stimuli input

    run_trial(congruent_input, initialize_input, gain)
    run_trial(neutral_input, initialize_input, gain)
    run_trial(incongruent_input, initialize_input, gain)
elif run_sim == 'gain_grid':
    for gain in gain_grid:
        initialize_input = trial_dict(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pc, 0.0, bias)
        # RUN TRIALS -------------------------------------------------------------------------------------------------
        congruent_input = trial_dict(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, pc, 0.0, bias)  # Create cong trial input
        neutral_input = trial_dict(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, pc, 0.0, bias)  # Create neutral trial stimuli input
        incongruent_input = trial_dict(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, pc, 0.0, bias)  # Create incong trial stimuli input

        run_trial(congruent_input, initialize_input, gain)
        run_trial(neutral_input, initialize_input, gain)
        run_trial(incongruent_input, initialize_input, gain)

# RESULTS --------------------------------------------------------------------------------------------------------------
"""
Per script a visualization is being made
"single" --> 9 graphs are visualized in a grid as done by Kalanthroff et al (2018) figure 2. For each combination of 
             condition and layer, the activation is shown for its units during the progress of the trial. 
"gain_grid" --> A visualization of gain (x-axis) versus RT (y-axis) for congruent, neutral and incongruent trials, as
                well as a visualization of gain (x-axis) versus facilitation/interference (y-axis).
                On top a pandas data-frame is made for further statistical analysis
"""
if run_sim == 'single':
    n_cong_RT = int((n_cong - settle) * slope + intercept)
    n_neutral_RT = int((n_neutral - settle) * slope + intercept)
    n_incong_RT = int((n_incong - settle) * slope + intercept)

    interference = n_incong_RT - n_neutral_RT
    facilitation = n_neutral_RT - n_cong_RT

    print("Congruency RT: " + str(n_cong_RT),
          "\nNeutral RT: " + str(n_neutral_RT),
          "\nIncongruency RT: " + str(n_incong_RT),
          "\nInterference: " + str(interference),
          "\nFacilitation: " + str(facilitation))

    print("Raw Congruency RT: " + str(n_cong),
          "\nRaw Neutral RT: " + str(n_neutral),
          "\nRaw Incongruency RT: " + str(n_incong))

    fig, axes = plt.subplots(nrows=3, ncols=4, sharey=True, sharex=True)
    axes[0, 0].set_ylabel('Congruent')
    axes[1, 0].set_ylabel('Neutral')
    axes[2, 0].set_ylabel('Incongruent')

    axes[0, 0].set_title('Task demand units', fontsize=9)
    axes[0, 1].set_title('Response units', fontsize=9)
    axes[0, 2].set_title('Color feature map', fontsize=9)
    axes[0, 3].set_title('Word feature map', fontsize=9)
    plt.setp(
        axes,
        xticks=[0, 400, 780],
        yticks=[0, 0.4, 0.79],
        yticklabels=['0', '0.4', '0.8'],
        xticklabels=['0', '400', '800']
    )

    plt.xlim(0, 800)
    plt.ylim(0, 0.8)

    # Plot congruent output --------------------------
    axes[0, 0].plot(ttt_cong[settle:, 0], 'c')
    axes[0, 0].plot(ttt_cong[settle:, 1], 'k')
    axes[0, 0].plot(conflict_cong, 'r')

    axes[0, 1].plot(rrr_cong[settle:, 0], 'b')
    axes[0, 1].plot(rrr_cong[settle:, 1], 'g')
    axes[0, 1].plot([0, n_cong - settle], [threshold, threshold], 'k')
    axes[0, 2].plot(ccc_cong[settle:, 0], 'b')
    axes[0, 2].plot(ccc_cong[settle:, 1], 'g')

    axes[0, 3].plot(www_cong[settle:, 0], 'b')
    axes[0, 3].plot(www_cong[settle:, 1], 'g')

    # Plot neutral output --------------------------
    axes[1, 0].plot(ttt_neutral[settle:, 0], 'c')
    axes[1, 0].plot(ttt_neutral[settle:, 1], 'k')
    axes[1, 0].plot(conflict_neutral, 'r')

    axes[1, 1].plot(rrr_neutral[settle:, 0], 'b')
    axes[1, 1].plot(rrr_neutral[settle:, 1], 'g')
    axes[1, 1].plot([0, n_neutral - settle], [threshold, threshold], 'k')
    axes[1, 2].plot(ccc_neutral[settle:, 0], 'b')
    axes[1, 2].plot(ccc_neutral[settle:, 1], 'g')

    axes[1, 3].plot(www_neutral[settle:, 0], 'b')
    axes[1, 3].plot(www_neutral[settle:, 1], 'g')

    # Plot incongruent output --------------------------
    axes[2, 0].plot(ttt_incong[settle:, 0], 'c')
    axes[2, 0].plot(ttt_incong[settle:, 1], 'k')
    axes[2, 0].plot(conflict_incong, 'r')

    axes[2, 1].plot(rrr_incong[settle:, 0], 'b')
    axes[2, 1].plot(rrr_incong[settle:, 1], 'g')
    axes[2, 1].plot([0, n_incong - settle], [threshold, threshold], 'k')
    axes[2, 2].plot(ccc_incong[settle:, 0], 'b')
    axes[2, 2].plot(ccc_incong[settle:, 1], 'g')

    axes[2, 3].plot(www_incong[settle:, 0], 'b')
    axes[2, 3].plot(www_incong[settle:, 1], 'g')

    plt.show()
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
    plt.rcParams['pdf.fonttype'] = 42 # To make fig illustrator compatable
    plt.rcParams['ps.fonttype'] = 42 # To make fig illustrator compatable

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
    plt.savefig("cong_incong_neutral_plot.pdf", transparent=True)
    plt.show()

    # Make gain facilitation/interference effect plot
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)  # Remove top edge
    ax.spines['right'].set_visible(False)  # Remove right edge
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_zorder(0)
    plt.plot(gain_grid, interference, 'b-', linewidth=4,
             label='Interference')
    plt.plot(gain_grid, facilitation, 'm--', linewidth=4,
             label='Facilitation')
    plt.xlabel('gain')
    plt.xticks(ticks=[0.95, 1.00, 1.05])
    plt.xlim(0.93, 1.07)
    plt.ylabel('Δ log RT(s)')
    plt.ylim(-0.05, 0.15)
    plt.yticks(ticks=[-0.05, 0.05, 0.15])
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()  # So that the labels don't fall out of the frame
    plt.savefig("fac_int_plot.pdf", transparent=True)
    plt.show()

    # Save files
    filename = "df_kalanthroff2018_gaingrid"
    sim_RT_df.to_csv(
        "{0}".format(filename),
        index=False)


