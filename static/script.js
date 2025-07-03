document.getElementById('predictForm').onsubmit = async function(event) {
    event.preventDefault();

    const data = {
      age: document.getElementById('age').value,
      weight: document.getElementById('weight').value,
      dosage: document.getElementById('dosage').value,
      reactions: document.getElementById('reactions').value,
      concomitant_drugs: document.getElementById('concomitant_drugs').value,
    };

    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    const resultData = await response.json();
    
    if (resultData.error) {
      alert(`Error: ${resultData.error}`);
      return;
    }

    document.getElementById('riskLevel').textContent =
      `Risk Level: ${resultData.risk_level} (${(resultData.prediction * 100).toFixed(2)}%)`;

    const shapPlotImg = document.getElementById('shapPlot');
    shapPlotImg.src = `data:image/png;base64,${resultData.shap_plot}`;
    shapPlotImg.style.display = "block";
};
