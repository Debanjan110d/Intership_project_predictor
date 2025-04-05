from django import template

register = template.Library()

@register.filter(name='addclass')
def addclass(value, arg):
    """Add CSS classes to form field"""
    if 'class' in value.field.widget.attrs:
        value.field.widget.attrs['class'] += f' {arg}'
    else:
        value.field.widget.attrs['class'] = arg
    return value
