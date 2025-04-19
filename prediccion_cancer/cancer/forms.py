from django import forms

class ImagenForm(forms.Form):
    imagen = forms.FileField(
        label='Selecciona una imagen (PNG o JPEG)',
        required=True,
        widget=forms.ClearableFileInput(attrs={'accept': 'image/png, image/jpeg'})
    )
