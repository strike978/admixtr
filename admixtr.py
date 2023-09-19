import streamlit as st
import numpy as np
import cvxpy as cp
from collections import defaultdict

# Initialize session state attributes
if 'textbox_content' not in st.session_state:
    st.session_state.textbox_content = ""

if 'target_textbox_content' not in st.session_state:
    st.session_state.target_textbox_content = ""

# Setting the layout of the page to wide and the title of the page to admixtr
st.set_page_config(layout="wide", page_title="admixtr", page_icon="🧬")
st.header(':red[admixtr]')


# Define the available data files
data_files = {
    "Modern Era": "Modern Ancestry.txt",
    "Mesolithic and Neolithic": "Mesolithic and Neolithic.txt",
    "Bronze Age": "Bronze Age.txt",
    "Iron Age": "Iron Age.txt",
    "Migration Period": "Migration Period.txt",
    "Middle Ages": "Middle Ages.txt",
}

# Define a function to read data from a file with UTF-8 encoding


def read_data_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def expand(pop_selector, pop_dict):
    pops = pop_selector.split('+')
    ret = []
    for pop in pops:
        ret += pop_dict.get(pop, pop)
    return ret


def distance(M, b):
    b = b[:, np.newaxis]
    return np.sqrt(np.sum((M - b) ** 2, axis=0))


# Create a multiselect checkbox to choose data files
selected_files = st.multiselect("Time Period:", list(
    data_files.keys()), default=["Modern Era"])

# Read data from selected files
selected_data = []
content_after_comma_set = set()
for file in selected_files:
    data = read_data_file(data_files[file])
    for line in data:
        content_after_comma = ",".join(line.split(',')[1:])
        if content_after_comma not in content_after_comma_set:
            selected_data.append(line)
            content_after_comma_set.add(content_after_comma)

# Get the populations already in the textbox
populations_in_textbox = [line.split(',')[1] if len(line.split(
    ',')) > 1 else '' for line in st.session_state.textbox_content.strip().split('\n')]

populations_in_target_textbox = [line.split(',')[1] if len(line.split(
    ',')) > 1 else '' for line in st.session_state.target_textbox_content.strip().split('\n')]

# Create a filtered list of available populations based on content after the comma
available_populations = [pop for pop in selected_data if pop.split(
    ',')[1] not in populations_in_textbox]

available_populations_target = [pop for pop in selected_data if pop.split(
    ',')[1] not in populations_in_target_textbox]


source_tab, target_tab, admixture_tab = st.tabs(
    ["Source", "Target", "Admixture"])

with source_tab:
    group_pop_toggle = st.toggle('Group Populations')

    # Group populations with the same word before the first ":" when toggle is enabled
    grouped_populations = {}
    if group_pop_toggle:
        for pop in available_populations:
            parts = pop.split(',')
            if len(parts) > 1:
                # Get the part before the first ":"
                key = parts[0].split(':')[0]
                if key not in grouped_populations:
                    grouped_populations[key] = []
                grouped_populations[key].append(pop)

    # Create a Selectbox to display populations based on the toggle
    if group_pop_toggle:
        population_options = list(grouped_populations.keys())
    else:
        population_options = available_populations

    selected_option_index = st.selectbox(
        "Populations:",
        range(len(population_options)),
        format_func=lambda i: population_options[i].split(',')[0]
    )

    # Determine the selected option based on the toggle
    if group_pop_toggle:
        selected_option = grouped_populations[population_options[selected_option_index]]
    else:
        selected_option = [population_options[selected_option_index]]

    # Create a button to add the selected option to the Textbox
    if st.button("Add Population"):
        if selected_option:
            for pop in selected_option:
                if pop not in st.session_state.textbox_content:
                    st.session_state.textbox_content += "\n" + pop
            st.experimental_rerun()

    # Display the Textbox with the entire selected options
    data_input = st.text_area('Enter data in G25 coordinates format:',
                              st.session_state.textbox_content.strip(), height=300, key='textbox_input')

    # Check if the Textbox content has changed manually and clear session state if it has
    if data_input != st.session_state.textbox_content.strip():
        st.session_state.textbox_content = data_input.strip()
        # Fixes issue with text reverting if changed twice?
        st.experimental_rerun()


with target_tab:
    selected_pop_index_target = st.selectbox("Populations:", range(len(
        available_populations_target)), format_func=lambda i: available_populations_target[i].split(',')[0], key="select_target_selectbox")
    if st.button("Add Population", key="target_select"):
        selected_population_target = available_populations_target[selected_pop_index_target]
        st.session_state.target_textbox_content = selected_population_target
        st.experimental_rerun()

    indivfile = st.text_input('Enter data in G25 coordinates format:',
                              st.session_state.target_textbox_content.strip(), key='text_input')

    # Check if the Textbox content has changed manually and clear session state if it has
    if indivfile != st.session_state.target_textbox_content.strip():
        st.session_state.target_textbox_content = indivfile.strip()
        st.experimental_rerun()


with admixture_tab:
    sheetfile = data_input

    ancestry_breakdown = []  # Initialize ancestry_breakdown outside the scope
    residual_norm = None  # Initialize residual_norm to None
    # Check if source and target text areas are empty
    source_empty = not st.session_state.textbox_content.strip()
    target_empty = not st.session_state.target_textbox_content.strip()

    col1, col2 = st.columns(2)

    aggregate = col1.toggle("Aggregate")

    reduce_populations = col2.toggle("Reduce Populations", disabled=True)
    # If the checkbox is toggled, add a number input for nonzeros
    if reduce_populations:
        nonzeros = st.number_input(
            "Number of Populations",
            min_value=3, max_value=5, value=5  # Set min to 3 and max to 5
        )
    else:
        nonzeros = 0  # Default value if checkbox is not selected

    calculation_button = False
    # Only show the "Calculate" button if both Source and Target are not empty

    if source_empty or target_empty:
        st.error("Please add populations to both Source and Target.")
    else:
        calculation_button = st.button("Calculate")
        if calculation_button:
            if sheetfile.strip() and indivfile.strip():
                # Your existing code starting from here
                m = []
                index2pop = []  # Store the order of population names
                index2indiv = []
                indiv2index = {}
                penalty = 0.
                threshold = .0001
                constraint_dict = {}
                operator_dict = {}
                pop_dict = defaultdict(list)
                # nonzeros = nonzeros

                # Process the pasted sheet file
                sheetfile_lines = sheetfile.splitlines()
                for line in sheetfile_lines:
                    arr = line.strip().split(',')
                    indivname = arr[0]
                    # Store population names in order
                    index2pop.append(indivname)
                    index2indiv.append(indivname)
                    indiv2index[indivname] = len(
                        index2indiv) - 1  # Assign an index
                    # Use the entire population name
                    pop_dict[indivname] = [indivname]
                    m.append(np.array([float(x) for x in arr[1:]]))

                M = np.column_stack(m)

                # Process the pasted indiv file
                indivfile_lines = indivfile.splitlines()
                for line in indivfile_lines:
                    arr = line.strip().split(',')
                    b = np.array([float(x) for x in arr[1:]])

                x = cp.Variable(M.shape[1])
                cost = cp.norm2(M @ x - b)**2 + penalty * \
                    cp.sum(cp.multiply(distance(M, b), x))

                constraints = [cp.sum(x) == 1, 0 <= x]

                for pop_selector, pen in constraint_dict.items():
                    op = operator_dict[pop_selector]
                    sum_expr = cp.sum([x[indiv2index[p]]
                                       for p in expand(pop_selector, pop_dict)])
                    if op == '=':
                        constraints.append(sum_expr == pen)
                    elif op == '>=':
                        constraints.append(sum_expr >= pen)
                    elif op == '<=':
                        constraints.append(sum_expr <= pen)

                if nonzeros > 0:
                    binary = cp.Variable(M.shape[1], boolean=True)
                    constraints += [x - binary <= 0.,
                                    cp.sum(binary) == nonzeros]

                prob = cp.Problem(cp.Minimize(cost), constraints)
                # prob.solve(cp.GUROBI)
                prob.solve()

                dindiv = defaultdict(int)
                dpop = defaultdict(int)

                for i, _ in enumerate(range(M.shape[1])):
                    dindiv[index2indiv[i]] += x.value[i]
                    dpop[index2pop[i]] += x.value[i]

                # Calculate ancestry breakdown
                ancestry_breakdown = [(k, v) for k, v in dindiv.items()]
                ancestry_breakdown.sort(key=lambda x: -x[1])

                # Calculate residual_norm within the calculation_button block
                residual_norm = cp.norm(M @ x - b, p=2).value
                target_name = indivfile.split(",")
                target_name = target_name[0]

                # Initialize a list to store the results
                results = []

                if residual_norm is not None:
                    fit_percentage_total = f"{residual_norm * 100:.4f}%"
                    results.append(
                        f'Target: {target_name} \nFit: {fit_percentage_total}')

                # Add "Aggregate" information
                if aggregate:
                    # Aggregate the ancestry breakdown
                    aggregated_ancestry = defaultdict(float)
                    for ancestry, percentage in ancestry_breakdown:
                        main_ancestry = ancestry.split(':')[0]
                        aggregated_ancestry[main_ancestry] += percentage
                    ancestry_breakdown = list(aggregated_ancestry.items())

                # Add ancestry breakdown
                for ancestry, percentage in ancestry_breakdown:
                    if percentage < threshold:
                        break
                    fit_percentage = f"{round(percentage * 100, 1)}%"
                    results.append(f'{fit_percentage} {ancestry}')

                # Display the results in a single code block
                if results:
                    st.code("\n\n".join(results))


# st.markdown(
#     "<span style='font-size: small;'>We extend our sincere gratitude to [michal3141](https://github.com/michal3141) for their generous contribution of the underlying source code that serves as the basis for this application. The original source code can be found [here](https://github.com/michal3141/g25).</span>", unsafe_allow_html=True)

st.caption(
    "Create Dendrograms and PCAs with [PopPlot](http://popplot.com)")
