# Purchase Intent Predictor

A machine learning powered Django web application that predicts customer purchase intent based on various behavioral and demographic features.



## Author
**Debanjan Dutta**

## Live Demo
🌐 [View Live Demo](https://debanjan2005.pythonanywhere.com/)

## Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python main.py`
4. Run Django server: `python manage.py runserver`

## Project Overview
This project uses machine learning to analyze customer behavior and predict their purchase intent. The model categorizes purchase intentions into four types:
- Impulsive
- Need-based
- Planned
- Wants-based

## Features
- Real-time purchase intent prediction
- Interactive web interface built with Django and Tailwind CSS
- Dark mode UI with modern glass-morphism design
- Probability distribution visualization for predictions
- Support for multiple customer attributes
- Responsive design for all devices

## Tech Stack
- **Backend**: Python, Django
- **Frontend**: HTML, Tailwind CSS
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib

## Project Structure
```bash
purchase_predictor/
├── data/
│   └── customer_data.csv
├── docs/
│   ├── model_training.md
│   └── django_deployment.md
├── predictor/
│   ├── models/
│   │   └── rf_model.joblib
│   └── train_model.py
├── web_interface/
│   ├── static/
│   └── templates/
├── requirements.txt
└── manage.py
```

## Installation
```bash
git clone https://github.com/yourusername/purchase-intent-predictor.git
cd purchase-intent-predictor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. Train the model:
```bash
python predictor/train_model.py
```

2. Start the Django server:
```bash
python manage.py runserver
```

3. Visit `http://localhost:8000` in your browser

## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Special thanks to my internship mentors
- Dataset provided by [source]
- Icons from [source]

📚 **Detailed Documentation:**
- [Model Training & Analysis](docs/model_training.md)
- [Django Development & Deployment](docs/django_deployment.md)
