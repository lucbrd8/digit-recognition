const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const predictButton = document.getElementById("predictButton");
const predictionResults = document.getElementById("predictionResults");

// Initialiser le canvas
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 10;
ctx.strokeStyle = "black";

let isDrawing = false;

// Fonctions pour dessiner
canvas.addEventListener("mousedown", () => {
    isDrawing = true;
});

canvas.addEventListener("mousemove", (e) => {
    if (!isDrawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mouseup", () => {
    isDrawing = false;
    ctx.beginPath();
});

canvas.addEventListener("mouseleave", () => {
    isDrawing = false;
    ctx.beginPath();
});

// Bouton pour effacer le canvas
function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Fonction pour envoyer l'image au backend et afficher les prédictions
predictButton.addEventListener("click", async () => {
    const imageData = canvas.toDataURL("image/png");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image: imageData }),
        });

        const predictions = await response.json();
        displayPredictions(predictions);
    } catch (error) {
        console.error("Erreur lors de la prédiction:", error);
    }
});

// Afficher les prédictions
function displayPredictions(predictions) {
    predictionResults.innerHTML = "";
    predictions.forEach((pred) => {
        const item = document.createElement("div");
        item.className = "prediction-item";
        item.innerHTML = `
            <strong>Chiffre ${pred.class}</strong>: ${(pred.probability * 100).toFixed(2)}%
            <div style="width: ${pred.probability * 100}%; background-color: #4CAF50; height: 10px; border-radius: 5px;"></div>
        `;
        predictionResults.appendChild(item);
    });
}

// Bouton pour effacer
const clearButton = document.createElement("button");
clearButton.textContent = "Effacer";
clearButton.addEventListener("click", clearCanvas);
document.body.appendChild(clearButton);