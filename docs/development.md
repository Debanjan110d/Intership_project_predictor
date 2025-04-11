# Development Guide
[← Back to README](../README.md)

## Setup Development Environment

### Prerequisites
- Python 3.8+
- Django 3.2+
- scikit-learn
- pandas
- numpy

### Installation
```bash
git clone <repository-url>
cd purchase_predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Development Server
```bash
cd purchase_predictor
python manage.py runserver
```

### Code Structure
- `predictor/views.py`: Main prediction logic
- `predictor/models.py`: Data models
- `predictor/forms.py`: Form definitions
- `templates/`: HTML templates

### Testing
```bash
python manage.py test predictor.tests
```

### Deployment
1. Update settings.py for production
2. Collect static files
3. Configure web server
4. Set up SSL certificate
