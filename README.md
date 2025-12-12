# ML-Lab2: Running Locally

This guide explains how to set up and run the ML-Lab2 project locally.

## Prerequisites

Ensure you have the following installed on your system:
- Python 3.8 or higher
- Node.js 16 or higher
- pnpm (Package Manager)
- pip (Python Package Installer)

## Steps to Run Locally

### 1. Clone the Repository
```bash
# Clone the repository
$ git clone https://github.com/lad94220/ML-Lab2.git
$ cd ML-Lab2
```

### 2. Set Up the Backend
```bash
# Navigate to the server directory
$ cd server

# Install Python dependencies
$ pip install -r requirements.txt

# Run the backend server
$ uvicorn main:app --reload
```

The backend server will be available at `http://127.0.0.1:8000`.

### 3. Set Up the Frontend
```bash
# Navigate to the client directory
$ cd ../client

# Install dependencies
$ pnpm install

# Run the development server
$ pnpm dev
```

Don't forget to create .env file with key `VITE_BACKEND_URI=http://localhost:8000`

The frontend will be available at `http://localhost:3000`.

### 4. Test the Application
- Open your browser and navigate to `http://localhost:3000`.
- Interact with the application to ensure everything is working correctly.

## Notes
- Ensure the backend server is running before starting the frontend.
- If you encounter issues, check the logs for both the backend and frontend for debugging.

## License
This project is licensed under the MIT License.