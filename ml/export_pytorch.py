"""
Export best_model.pt (PyTorch) → best_model.onnx (ONNX Runtime compatible).

Le modèle PyTorch est chargé depuis ml/checkpoints/best_model.pt.
L'ONNX exporté est utilisé par ConsumerApp (ModelLoader.scala, priorité maximale).

Usage :
    python ml/export_pytorch.py \
        --checkpoint ml/checkpoints/best_model.pt \
        --output     ml/checkpoints/best_model.onnx \
        --img-size   256 \
        --num-classes 2

Puis copier vers HDFS pour que le Consumer le charge :
    hdfs dfs -put ml/checkpoints/best_model.onnx hdfs:///project/models/

Conventions ONNX attendues par ModelLoader.scala :
    - input  name : "image_input"  shape [batch, 3, img_size, img_size]  dtype float32
    - output name : "output"       shape [batch, num_classes]             dtype float32
    - normalisation ImageNet : mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

Dépendances :
    pip install torch onnx onnxruntime
"""

import argparse
import sys
import os

import torch
import torch.nn as nn
import onnx
import onnxruntime as rt
import numpy as np


def load_model(checkpoint_path: str, num_classes: int) -> nn.Module:
    """
    Tente de charger best_model.pt dans l'ordre :
      1. Objet nn.Module complet (torch.save(model, path))
      2. Dict avec clé 'model'
      3. Dict avec clé 'model_state_dict' → enveloppe dans un wrapper générique
    """
    # weights_only=True : un checkpoint pickle peut exécuter du code arbitraire
    # au chargement. Le déballage complet (nn.Module picklé) n'est accepté que
    # si ALLOW_UNSAFE_CHECKPOINT=1 est posé explicitement (source de confiance).
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        if os.environ.get("ALLOW_UNSAFE_CHECKPOINT") != "1":
            raise SystemExit(
                f"[Export] {checkpoint_path} n'est pas un state_dict pur "
                "(weights_only=True a échoué). Si la source est de confiance, "
                "relancer avec ALLOW_UNSAFE_CHECKPOINT=1.")
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(raw, nn.Module):
        print(f"[Export] Chargé comme nn.Module : {type(raw).__name__}")
        return raw.eval()

    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], nn.Module):
            print(f"[Export] Extrait raw['model'] : {type(raw['model']).__name__}")
            return raw["model"].eval()

        if "model_state_dict" in raw or "state_dict" in raw:
            key = "model_state_dict" if "model_state_dict" in raw else "state_dict"
            state_dict = raw[key]
            print(f"[Export] state_dict détecté ({len(state_dict)} clés)")
            print("[Export] Architecture inconnue — tentative avec ResNet-18 wrapper.")
            print("[Export] Si l'export échoue, définissez votre architecture dans ce script.")

            try:
                import torchvision.models as models
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(state_dict, strict=False)
                print("[Export] ResNet-18 chargé (strict=False — vérifiez la précision)")
                return model.eval()
            except Exception as e:
                print(f"[Export] ERREUR chargement ResNet-18 : {e}")
                print("[Export] Définissez manuellement votre architecture dans ce script.")
                sys.exit(1)

    raise ValueError(f"Format inconnu : {type(raw)}")


def export_onnx(model: nn.Module, output_path: str, img_size: int) -> None:
    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=11,
        input_names=["image_input"],
        output_names=["output"],
        dynamic_axes={
            "image_input": {0: "batch_size"},
            "output":      {0: "batch_size"},
        },
    )
    print(f"[Export] ONNX écrit : {output_path}")


def verify_onnx(onnx_path: str, img_size: int) -> None:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[Export] Vérification ONNX : OK")

    sess = rt.InferenceSession(onnx_path)
    inp  = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    out  = sess.run(None, {"image_input": inp})
    print(f"[Export] Inférence test — output shape : {out[0].shape}")
    print(f"[Export] Inputs  ONNX : {[i.name for i in sess.get_inputs()]}")
    print(f"[Export] Outputs ONNX : {[o.name for o in sess.get_outputs()]}")
    print("[Export] Export validé. Copier vers hdfs:///project/models/best_model.onnx")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="ml/checkpoints/best_model.pt")
    parser.add_argument("--output",      default="ml/checkpoints/best_model.onnx")
    parser.add_argument("--img-size",    type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Nombre de classes du modèle (défaut 2 : background + plate)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[Export] Checkpoint introuvable : {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint, args.num_classes)
    export_onnx(model, args.output, args.img_size)
    verify_onnx(args.output, args.img_size)


if __name__ == "__main__":
    main()
