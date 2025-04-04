from django import forms

class PredictionForm(forms.Form):
    # Numeric fields with ranges
    age = forms.FloatField(
        min_value=18, max_value=50,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    purchase_amount = forms.FloatField(
        min_value=50.0, max_value=500.0,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    time_spent = forms.FloatField(
        label='Time Spent on Research (hours)',
        min_value=0, max_value=2.5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    # Categorical fields with choices
    GENDER_CHOICES = [
        ('Female', 'Female'), ('Male', 'Male'), 
        ('Bigender', 'Bigender'), ('Agender', 'Agender'),
        ('Genderfluid', 'Genderfluid'), ('Non-binary', 'Non-binary'),
        ('Polygender', 'Polygender')
    ]
    
    INCOME_CHOICES = [
        ('High', 'High'), ('Middle', 'Middle'), ('Low', 'Low')
    ]
    
    PURCHASE_CATEGORY_CHOICES = [
        ('Electronics', 'Electronics'), ('Sports & Outdoors', 'Sports & Outdoors'),
        ('Jewelry & Accessories', 'Jewelry & Accessories'), ('Home Appliances', 'Home Appliances'),
        ('Toys & Games', 'Toys & Games'), ('Mobile Accessories', 'Mobile Accessories'),
        ('Food & Beverages', 'Food & Beverages'), ('Beauty & Personal Care', 'Beauty & Personal Care'),
        ('Books', 'Books'), ('Furniture', 'Furniture'), ('Health Care', 'Health Care'),
        ('Gardening & Outdoors', 'Gardening & Outdoors'), ('Clothing', 'Clothing'),
        ('Luxury Goods', 'Luxury Goods'), ('Office Supplies', 'Office Supplies'),
        ('Arts & Crafts', 'Arts & Crafts'), ('Baby Products', 'Baby Products'),
        ('Animal Feed', 'Animal Feed'), ('Travel & Leisure (Flights)', 'Travel & Leisure (Flights)'),
        ('Hotels', 'Hotels'), ('Health Supplements', 'Health Supplements'),
        ('Software & Apps', 'Software & Apps'), ('Groceries', 'Groceries'),
        ('Packages', 'Packages')
    ]
    
    DEVICE_CHOICES = [
        ('Desktop', 'Desktop'), ('Tablet', 'Tablet'), ('Smartphone', 'Smartphone')
    ]

    gender = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    income_level = forms.ChoiceField(choices=INCOME_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    purchase_category = forms.ChoiceField(choices=PURCHASE_CATEGORY_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    device = forms.ChoiceField(choices=DEVICE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    
    # Boolean fields
    discount_used = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}))
    loyalty_member = forms.BooleanField(required=False, widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}))
