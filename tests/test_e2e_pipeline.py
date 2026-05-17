"""End-to-end pipeline test for both models."""
import torch
from model_service.inference.orchestrator import ModelRegistry, InferenceOrchestrator
from model_service.acne_model.model import AcneSeverityModel
from model_service.pores_model.model import PoreSeverityModel

# Setup registry with both models
registry = ModelRegistry()

# Register acne (no real weights, just structure)
acne_model = AcneSeverityModel(pretrained=False)
acne_model.eval()
registry._models["acne"] = acne_model.to(registry.device)

# Register pores with trained weights
registry.register_pores_model("model_service/checkpoints/pores_model_best.pth")

print(f"Loaded models: {registry.loaded_models}")

# Run both predictions on a dummy image
orchestrator = InferenceOrchestrator(registry)
dummy_tensor = torch.randn(3, 224, 224)
predictions = orchestrator.predict_all(dummy_tensor)

acne_p = predictions["acne"]
pores_p = predictions["pores"]
print(f"Acne prediction: {acne_p.predicted_label} (conf={acne_p.confidence})")
print(f"Pores prediction: {pores_p.predicted_label} (conf={pores_p.confidence})")

# Run through BLP
from backend.blp.engine import BLPEngine
engine = BLPEngine()
result = engine.process(predictions)

print(f"BLP - Acne severity: {result.acne_severity}")
print(f"BLP - Pore severity: {result.pore_severity}")
print(f"BLP - Recommendations: {len(result.recommendations)} items")

# Build report
from backend.report_generation.builder import ReportBuilder
report = ReportBuilder.build_success_report(predictions, result)
print(f"Report ID: {report.id}")
print(f"Report status: {report.status}")
print(f"Report acne_severity: {report.acne_severity}")
print(f"Report pore_severity: {report.pore_severity}")
print(f"Report recommendations: {len(report.recommendations)}")
print("FULL PIPELINE OK")
