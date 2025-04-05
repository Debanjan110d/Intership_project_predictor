# Django Development & Deployment Guide
[← Back to README](../README.md) | [Model Training Documentation ←](model_training.md)

## Overview
This guide details the Django implementation and deployment process for the Purchase Intent Predictor.

## Project Structure
![Project Structure](images/project_structure.png)

## Implementation Steps
1. **Setup Django Project**
```bash
django-admin startproject purchase_predictor
cd purchase_predictor
python manage.py startapp predictor
```

2. **Configure Settings**
```python
INSTALLED_APPS = [
    ...
    'predictor',
]
```

3. **Create Models Directory**
```bash
mkdir predictor/models
touch predictor/models/__init__.py
```

## Form Implementation
![Form Interface](images/form.png)
*Prediction input form with validation*

### Key Components
- Custom form validation
- Tailwind CSS styling
- Mobile responsiveness

## Results Display
![Results Page](images/results.png)
*Interactive results visualization*

## Deployment Steps

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver
```

### PythonAnywhere Deployment
1. Create PythonAnywhere account
2. Upload project files
3. Set up virtual environment
4. Configure WSGI file
5. Update settings.py
6. Reload web app

## Monitoring
- Check error logs
- Monitor performance
- Track usage statistics

## Related Documentation
- [Project README](../README.md)
- [Model Training Guide](model_training.md)
