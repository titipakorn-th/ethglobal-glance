FROM tiangolo/uvicorn-gunicorn:python3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY chatbot ./chatbot

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]