from django.shortcuts import render
from .forms import ImagenForm

# Create your views here.

from django.http import HttpResponse

def index(request):
    return render(request, "index.html")

def subir_imagen(request):
    if request.method == 'POST' and request.FILES['imagen']:
        form = ImagenForm(request.POST, request.FILES)
        if form.is_valid():
            # Aquí puedes manejar la imagen, guardarla o procesarla
            imagen = form.cleaned_data['imagen']
            # Guardar o hacer algo con la imagen
            # ejemplo: imagen.save()
    else:
        form = ImagenForm()
    
    return render(request, 'subir_imagen.html', {'form': form})