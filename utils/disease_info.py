# disease_info.py

DISEASE_KNOWLEDGE_BASE = {
    "Healthy": {
        "symptoms": "No visible symptoms. The leaf appears green and healthy.",
        "treatment_suggestion": "No treatment required.",
        "prevention_tips": "Continue regular watering, fertilization, and monitoring."
    },
    "Apple Scab": {
        "symptoms": "Olive-green to black spots on leaves, fruit, and petioles.",
        "treatment_suggestion": "Apply a suitable fungicide (e.g., captan, myclobutanil) as per agricultural guidelines.",
        "prevention_tips": "Rake and destroy fallen leaves. Ensure good air circulation by pruning."
    },
    "Apple Black Rot": {
        "symptoms": "Brown lesions on leaves (frog-eye leaf spot), black rotted fruit.",
        "treatment_suggestion": "Remove infected plant parts. Apply preventive fungicides.",
        "prevention_tips": "Remove dead wood and mummified fruit. Improve tree vigor."
    },
    "Apple Cedar Rust": {
        "symptoms": "Yellow-orange spots on leaves and fruit, often with a rusty appearance.",
        "treatment_suggestion": "Use fungicides labeled for rust diseases applied during the susceptible period.",
        "prevention_tips": "Remove nearby alternate hosts (like cedar or juniper trees)."
    },
    "Corn Common Rust": {
        "symptoms": "Oval to elongate, cinnamon-brown pustules scattered on both surfaces of the leaves.",
        "treatment_suggestion": "Apply fungicide if the infection is severe and environmental conditions favor disease.",
        "prevention_tips": "Plant rust-resistant corn varieties."
    },
    "Tomato Blight": {
        "symptoms": "Dark, concentric ring spots on older leaves, wilting and fruit rot.",
        "treatment_suggestion": "Apply fungicides like chlorothalonil or copper-based sprays.",
        "prevention_tips": "Avoid overhead watering, practice crop rotation, and remove affected leaves early."
    },
    "Unknown Disease": {
        "symptoms": "Symptoms typical for general stress or unclassified ailment.",
        "treatment_suggestion": "Consult a local agricultural expert or extension office.",
        "prevention_tips": "Maintain standard crop hygiene and optimal growing conditions."
    }
}

def get_disease_info(disease_name):
    """
    Returns the knowledge base information for a given disease name.
    """
    return DISEASE_KNOWLEDGE_BASE.get(disease_name, DISEASE_KNOWLEDGE_BASE["Unknown Disease"])
