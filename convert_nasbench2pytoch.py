import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import sys
from thop import profile
sys.path.append("src/nas2model")

def export_pytorch(model, output_path='model.pth'):
    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model_cpu, dummy_input)
    torch.jit.save(traced_model, output_path)
    print(f"Saved TorchScript model to: {output_path}")

def evaluate(model, loader, criterion, device):
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description="Train a basic CNN on CIFAR-10 and export to ONNX/RKNN")
    parser.add_argument("--nasbench_type", type=str, default="nasbench101", help="Type of NASBench dataset (e.g., nasbench101)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="CIFAR-10 data directory")
    parser.add_argument("--json_folder", type=str, default="topk_nasbench_outputs/nasbench101", help="Folder containing NASBench101 genotype JSON files")
    parser.add_argument("--folder_output", type=str, default="topk_nasbench_models/pytorch/nasbench101", help="Folder to save exported models")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model = BasicCNN(num_classes=10).to(device)
    

    # Example genotype (5 intermediate nodes) for test
    def nasbench2model(json_file, folder_output='output_models'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hash_id = os.path.basename(json_file).split(".")[0].split("-")[-1]
        output_path=f'{folder_output}/{hash_id}_{args.epochs}epochs_cifar10.pth'
        if os.path.exists(output_path):
            print(f"Model for {hash_id} already exists, skipping...")
            return
        import json
        genotype = json.load(open(json_file))

        if args.nasbench_type == "nasbench101":
            from nasbench101 import Network
            matrix = genotype["adjacency_matrix"]
            ops = genotype["operations"]
            model = Network(matrix, ops).to(device)
        elif args.nasbench_type == "nasbench201":
            from nasbench201 import build_nasbench201_model_from_string
            arch_str = genotype["arch"]
            model = build_nasbench201_model_from_string(arch_str=arch_str, C=16, num_classes=10).to(device)
        elif args.nasbench_type == "nasbench301":
            from nasbench301 import Network, json_to_genotype
            geno_dict = genotype['genotype']
            genotype = json_to_genotype(geno_dict)
            model = Network(C=36, num_classes=10, layers=8, genotype=genotype).to(device)
        elif args.nasbench_type == "hwnasbench":
            from nasbench201 import build_nasbench201_model_from_string
            arch_str = genotype["arch_str"]
            model = build_nasbench201_model_from_string(arch_str=arch_str, C=16, num_classes=10).to(device)
        else:
            raise ValueError(f"Unsupported NASBench type: {args.nasbench_type}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total if total > 0 else 0.0
            train_acc = 100.0 * correct / total if total > 0 else 0.0

            print(
                f"epoch {epoch + 1}/{args.epochs} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.2f}% | "
            )

        model.eval()
        
        os.makedirs(folder_output, exist_ok=True)
        export_pytorch(model, output_path=output_path)

        print(f"Model for {hash_id} trained and saved to {output_path}")
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        dummy_input = torch.randn(1, 3, 32, 32).to(device)  # adjust input size if needed

        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        metrics = {
            "hash_id": hash_id,
            "flops": int(flops),
            "total_params": int(params),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epochs": args.epochs
        }


        # print(f"Accuracy on test set: {val_acc:.2f}%")
        # device = torch.device("cpu")
        # model = torch.jit.load(output_path, map_location=device)
        # model.eval()
        # val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        # print(f"Re-loaded model test accuracy: {val_acc:.2f}%")

        metrics_output_path = os.path.join(folder_output, "metrics")
        os.makedirs(metrics_output_path, exist_ok=True)
        with open(os.path.join(metrics_output_path, f"{hash_id}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)


    for json_file in os.listdir(args.json_folder):
        if json_file.endswith(".json"):
            print(f"Processing {json_file}...")
            nasbench2model(os.path.join(args.json_folder, json_file), folder_output=args.folder_output)

if __name__ == "__main__":
    main()

# Example usage:
# cd /home/jupyter-hunglk/projects/nas-hoi/Nas2ModelTool
# python main.py --epochs 36 --json_folder ../results/nasbench101 --folder_output results/nasbench101 --nasbench_type nasbench101
# python main.py --epochs 36 --json_folder ../results/nasbench201 --folder_output results/nasbench201 --nasbench_type nasbench201
# python main.py --epochs 36 --json_folder ../results/nasbench301 --folder_output results/nasbench301 --nasbench_type nasbench301
# python main.py --epochs 36 --json_folder ../results/hwnasbench --folder_output results/hwnasbench --nasbench_type hwnasbench