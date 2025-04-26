document.addEventListener('DOMContentLoaded', function() {
  const formulario = document.getElementById('formulario');
  
  // Mostrar el valor del slider al lado
  const slider = document.getElementById('image_weight');
  const output = document.getElementById('weight_value');
  output.textContent = 'Peso de la imagen: ' + slider.value + '%';
  slider.oninput = function() {
      output.textContent = 'Peso de la imagen: ' + slider.value + '%';
  };

  formulario.addEventListener('submit', function(e) {
    e.preventDefault();  // Prevenir el envío normal del formulario

    // Crear un objeto FormData para recoger todos los campos, incluyendo la imagen
    const formData = new FormData(formulario);

    // Obtener el archivo de la imagen
    const imagen = document.getElementById('id_imagen').files[0];

    // Si hay una imagen, convertirla a base64
    if (imagen) {
      const reader = new FileReader();
      reader.onloadend = function() {
        // Una vez que la imagen se ha convertido a base64, recoger el valor
        const imagenBase64 = reader.result;

        // Recoger el resto de los datos del formulario
        console.log(slider.value);
        const datos = {
          Sexo: document.getElementById('Sexo').value,
          Age: document.getElementById('Age').value,
          'Family history': document.getElementById('Family history').checked ? 'Yes' : 'No',
          smoke: document.getElementById('smoke').checked ? 'Yes' : 'No',
          alcohol: document.getElementById('alcohol').checked ? 'Yes' : 'No',
          obesity: document.getElementById('obesity').value,
          diet: document.getElementById('diet').value,
          Screening_History: document.getElementById('Screening_History').value,
          Healthcare_Access: document.getElementById('Healthcare_Access').value,
          cancer_stage: document.getElementById('cancer_stage').value,
          tumor_size: document.getElementById('tumor_size').value,
          early_detection: document.getElementById('early_detection').checked ? 'Yes' : 'No',
          inflammatory_bowel_disease: document.getElementById('inflammatory_bowel_disease').checked ? 'Yes' : 'No',
          relapse: document.getElementById('relapse').checked ? 'Yes' : 'No',
          imagen_base64: imagenBase64,  // Añadimos la imagen convertida a base64
          image_weight: slider.value   // Recogemos el valor del slider
        };

        // Usar fetch para enviar los datos al servidor
        fetch('/predict/', {  // La URL de la vista
          method: 'POST',
          headers: {
            "X-CSRFToken": getCookie("csrftoken"),
            "Content-Type": "application/json",
          },
          body: JSON.stringify(datos),  // Enviamos los datos como JSON
        })
        .then(response => response.json())  // Convertimos la respuesta en JSON
        .then(data => {
          // Verificamos si la predicción es 1 (maligno) o 0 (benigno)
          if (data.prediction === 1) {
            // Si la predicción es 1, mostrar "Cáncer Maligno"
            document.getElementById('prediccion').textContent = "Cáncer Maligno";
          } else if (data.prediction === 0) {
            // Si la predicción es 0, mostrar "Cáncer Benigno"
            document.getElementById('prediccion').textContent = "Cáncer Benigno";
          } else {
            alert('Error en la predicción.');
          }
          document.getElementById('resultados').style.display = 'block';
        })
        .catch(error => {
          console.error('Error:', error);
          alert('Hubo un error al procesar la solicitud.');
        });
      };
      reader.readAsDataURL(imagen);  // Convertir la imagen a base64
    } else {
      alert("Por favor, sube una imagen.");
    }
  });

  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
        const cookies = document.cookie.split(";");
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === name + "=") {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
  }
});
