import streamlit as st
import requests

# Set the title of the app
st.title("Content Moderation and Translation")

# Text area for user input
message = st.text_area("Enter your message:")

# Create two columns for buttons
col1, col2 = st.columns(2)

# Check button
with col1:
    if st.button("Check"):
        try:
            # Send a request to the moderate endpoint
            response = requests.post("http://localhost:3000/moderate/", json={"text": message})
            response.raise_for_status()  # Raise an error for bad responses

            # Display the moderation result
            result = response.json()
            if result["prediction"]["overall"] == "Appropriate":
                st.success("Message is appropriate.")
            else:
                st.error("Message is inappropriate.")
                # Display specific labels that made the text inappropriate
                st.write("Inappropriate labels:")
                for label, value in result["prediction"].items():
                    if value == "Yes" and label != "overall":
                        st.write(f"- {label}")
        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error: {errh}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
        except ValueError:
            st.error("Received non-JSON response from server.")

# Translate button
with col2:
    if st.button("Translate"):
        try:
            # Send a request to the translate endpoint
            response = requests.post("http://localhost:3000/translate/", json={"text": message})
            response.raise_for_status()  # Raise an error for bad responses

            # Display the translation result
            translated_text = response.json()["response"]
            st.success(f"Translated text: {translated_text}")
        except requests.exceptions.HTTPError as errh:
            st.error(f"HTTP Error: {errh}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
        except ValueError:
            st.error("Received non-JSON response from server.")