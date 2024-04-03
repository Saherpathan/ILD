import streamlit as st

# Deep Learning roadmap structure (feel free to customize)
deep_learning_roadmap = {
    "Fundamentals": [
        "Linear Algebra",
        "Calculus",
        "Probability & Statistics",
        "Python Programming",
        "Machine Learning Basics"
    ],
    "Deep Learning Concepts": [
        "Artificial Neural Networks (ANNs)",
        "Activation Functions",
        "Loss Functions",
        "Optimization Algorithms (Gradient Descent, etc.)",
        "Backpropagation"
    ],
    "Deep Learning Architectures": [
        "Convolutional Neural Networks (CNNs)",
        "Recurrent Neural Networks (RNNs)",
        "Long Short-Term Memory (LSTM)",
        "Transformers",
        "Autoencoders",
        "Generative Adversarial Networks (GANs)"
    ],
    "Applications": [
        "Computer Vision (Image Recognition, Object Detection)",
        "Natural Language Processing (Text Classification, Machine Translation)",
        "Speech Recognition",
        "Reinforcement Learning",
        "Time Series Forecasting"
    ],
    "Advanced Topics": [
        "Regularization Techniques",
        "Deep Learning Frameworks (TensorFlow, PyTorch)",
        "Hyperparameter Tuning",
        "Explainable AI (XAI)",
        "Research in Deep Learning"
    ]
}

# User input for personalization
user_query = st.text_input("Enter your query (e.g., 'Deep Learning for Computer Vision')")
user_query = user_query.lower()  # Case-insensitive matching

# Display roadmap based on user input (optional filtering)
st.title("Deep Learning Roadmap")

# Option 1: Highlight relevant sections (basic approach)
relevant_sections = []
for section, topics in deep_learning_roadmap.items():
    if any(query_term in topic.lower() for query_term in user_query.split()):
        relevant_sections.append(section)

if relevant_sections:
    st.write("Here's a roadmap tailored to your query:")
    for section in relevant_sections:
        with st.expander(section):
            st.write("\n".join(deep_learning_roadmap[section]))
else:
    st.write("Your query didn't match any specific sections. Here's the full roadmap for reference:")
    for section, topics in deep_learning_roadmap.items():
        with st.expander(section):
            st.write("\n".join(topics))

# Option 2: More sophisticated filtering and prioritization (requires more customization)
# You could implement a scoring system based on keyword matches,
# topic relevance to the query, or user's background (if provided)
# to prioritize sections and topics in the displayed roadmap.

# Additional resources
st.header("Additional Resources")
st.write("Here are some helpful resources to complement your Deep Learning learning path:")
st.write("- [Deep Learning Book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)")
st.write("- [Stanford's CS231n: Convolutional Neural Networks for Visual Recognition course](http://cs231n.stanford.edu/)")
st.write("- [DeepMind's Deep Learning course](https://deepmind.com/deep-learning/course/)")
st.write("- [The TensorFlow tutorials](https://www.tensorflow.org/tutorials/)")
st.write("- [The PyTorch tutorials](https://pytorch.org/tutorials/)")

# Personalization section (optional)
st.header("Personalize Your Roadmap")
st.write("This roadmap is a starting point. Feel free to adjust it based on your background, interests, and goals. Here are some prompts to guide you:")
st.write("- What specific applications of Deep Learning are you most interested in?")
st.write("- What is your current level of programming experience")
