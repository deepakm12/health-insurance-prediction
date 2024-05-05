from src.interface import HealthInsuranceApp
import streamlit as st
import pandas as pd


def run():
    """Main function to run the program."""
    st.set_page_config(page_title="Health Insurance Cost Predictor", layout="wide")
    st.title("Health Insurance Cost Predictor")

    # Sidebar options
    st.sidebar.header("Options")
    app = HealthInsuranceApp()

    # Re-train models button
    if st.sidebar.button("Re-Train Models"):
        with st.spinner("Training models..."):
            app.train_models()
        st.success("Models re-trained successfully!")

    # Visualize data button
    if st.sidebar.button("Visualize Data"):
        st.header("Visualization of Data")
        plots = app.visualize_data()
        for plot in plots:
            st.plotly_chart(plot, use_container_width=True)

    # Visualize model results button
    if st.sidebar.button("Visualize Model Results"):
        st.header("Visualization of Model Results")
        plots = app.visualize_results()
        for plot in plots:
            st.plotly_chart(plot, use_container_width=True)

    # Visualize model results button
    if st.sidebar.button("See Dataset"):
        st.header("Dataset")
        st.dataframe(app.data())

    # Input data for prediction
    st.sidebar.header("Input Data for Prediction")
    input_data = app.input_data()

    # Select model for prediction
    model = st.sidebar.selectbox(
        "Select Model",
        ["RandomForest", "KNeighbors", "Linear", "Ridge", "SVR"],
        index=0,
    )

    # Predict medical costs button
    if st.sidebar.button("Predict Medical Costs"):
        with st.spinner("Predicting medical costs..."):
            prediction = app.predict_medical_costs(input_data, model)
        if prediction is not None:
            st.subheader("Predicted Medical Costs")
            st.success(f"The predicted medical cost is: {prediction}")

    st.markdown("#### 👨‍💼 Contributing & Licence")
    st.markdown(
        "Contributions are welcome! 🎉 Please feel free to raise an issue or submit a PR. 😊"
    )
    st.markdown(
        "This project is under license from MIT. For more details, see the [LICENSE](https://github.com/deepakm12/health-insurance-prediction/blob/main/LICENSE) file."
    )
    st.markdown("Made by [Deepak Mahto](https://github.com/deepakm12)")


if __name__ == "__main__":
    run()
