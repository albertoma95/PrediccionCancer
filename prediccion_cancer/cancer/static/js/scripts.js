
  document.addEventListener('DOMContentLoaded', function() {
    const formulario = document.getElementById('formulario');
    
    formulario.addEventListener('submit', function(e) {
      e.preventDefault();  // Prevenir el envío normal del formulario


      // Recoger los datos de los campos de texto y checkbox
      const datos = {
        Sexo: document.getElementById('Sexo').value,  // Recoger el valor del select
        Age: document.getElementById('Age').value,  // Recoger el valor de la edad
        'Family history': document.getElementById('Family history').checked? 'Yes' : 'No'? 'Yes' : 'No',  // Recoger el valor de Family history
        smoke: document.getElementById('smoke').checked? 'Yes' : 'No',  // Recoger el estado del checkbox (true o false)
        alcohol: document.getElementById('alcohol').checked? 'Yes' : 'No',  // Recoger el estado del checkbox (true o false)
        obesity: document.getElementById('obesity').value,  // Recoger el valor del select de obesidad
        diet: document.getElementById('diet').value,  // Recoger el valor de diet
        Screening_History: document.getElementById('Screening_History').value,  // Recoger el valor de Screening_History
        Healthcare_Access: document.getElementById('Healthcare_Access').value,  // Recoger el valor de Healthcare_Access
        cancer_stage: document.getElementById('cancer_stage').value,  // Recoger el valor de cancer_stage
        tumor_size: document.getElementById('tumor_size').value,  // Recoger el valor de tumor_size
        early_detection: document.getElementById('early_detection').checked? 'Yes' : 'No',  // Recoger el valor de early_detection
        inflammatory_bowel_disease: document.getElementById('inflammatory_bowel_disease').checked? 'Yes' : 'No',  // Recoger el estado del checkbox
        relapse: document.getElementById('relapse').checked? 'Yes' : 'No'
      };


      // Usar fetch para enviar los datos al servidor
      fetch('/predict/', {  // La URL de la vista
        method: 'POST',
        headers: {
          "X-CSRFToken": getCookie("csrftoken"),
          "Content-Type": "application/json",
        },
        body: JSON.stringify(datos),
      })
      .then(response => response.json())  // Convertimos la respuesta en JSON
      .then(data => {
        if (data.prediction) {
          // Si la predicción se recibió, mostrarla
          document.getElementById('prediccion').textContent = data.prediction;
          document.getElementById('resultados').style.display = 'block';
        } else {
          alert('Error en la predicción.');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Hubo un error al procesar la solicitud.');
      });
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