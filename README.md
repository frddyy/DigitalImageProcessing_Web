# Digital Image Processing Web Application

This is a web-based application that allows users to upload images and apply various digital image processing features. The application is built using Python and Flask.

## Getting Started

Follow the steps below to set up and run the application locally.

### 1. Clone the Repository

First, clone this repository to your local machine:
```bash
git clone https://github.com/frddyy/DigitalImageProcessing_Web.git
cd DigitalImageProcessing_Web
```

### 2. Set Up Virtual Environment

Create a virtual environment to manage dependencies:
```bash
python -m venv .venv
```

Activate the virtual environment:
- On Windows:
  ```bash
  .venv\Scripts\activate
  ```
- On Linux/Mac:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

Install the required dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Flask server:
```bash
python app.py
```

The application will run on `http://127.0.0.1:5000`. Open this URL in your web browser to access the interface.

### 5. File Structure

The project folder contains the following structure:
```
.
├── .venv/               # Virtual environment
├── static/              # Static files (CSS, images, etc.)
├── templates/           # HTML templates
├── app.py               # Main Flask application
├── image_processing.py  # Image processing functions
├── requirements.txt     # Dependencies
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

### 6. Notes

- Ensure that the `requirements.txt` file includes all necessary dependencies for the application.
- Use the `.gitignore` file to exclude unnecessary files and folders (e.g., `.venv`, `__pycache__`).

Feel free to modify and expand this application as needed!
