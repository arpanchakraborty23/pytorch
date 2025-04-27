from data_download import download_data
from dataset_dataloader import create_dataloader
from model import TinyVGG 
from train import Train
from utils import save_model
import torch
from torchvision import transforms
from torchinfo import summary


class FoodImageClassifier:
    def __init__(self,
                 url: str,
                 hidden_units: int = 12,
                 image_size: int = 64,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 epochs: int = 5,
                 device: str = None):
        """
        Class for training a TinyVGG model on a food image dataset.

        Args:
            url (str): URL for dataset download.
            hidden_units (int): Number of hidden units in TinyVGG.
            image_size (int): Size to which images are resized.
            batch_size (int): Batch size for dataloaders.
            learning_rate (float): Learning rate for optimizer.
            epochs (int): Number of training epochs.
            device (str, optional): "cuda" or "cpu". Auto-detects if None.
        """
        self.url = url
        self.hidden_units = hidden_units
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        # Step 1: Download data
        self.data_path, self.train_dir, self.test_dir = download_data(url=self.url)
        print('Data Download Completed')
        print('==============================')

        # Step 2: Setup transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size=(self.image_size, self.image_size)),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Step 3: Create dataloaders
        self.train_loader, self.test_loader, self.num_classes = create_dataloader(
            train_dir=self.train_dir,
            test_dir=self.test_dir,
            transform=self.transforms,
            batch_size=self.batch_size,
        )
        print('Data Transformation and DataLoader Completed')
        print('==============================')

        # Step 4: Build model
        self.model = TinyVGG(
            input_shape=3,
            hidden_units=self.hidden_units,
            output_shape=len(self.num_classes)
        ).to(self.device)
        print('TinyVGG Model Summary')
        print(summary(self.model, input_size=[self.batch_size, 3, self.image_size, self.image_size]))
        print('==============================')

        # Step 5: Setup optimizer and loss
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        # Step 6: Train model
        self.trainer = Train(
            model=self.model,
            epochs=self.epochs,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            train_dataloader=self.train_loader,
            test_dataloader=self.test_loader,
            device=self.device
        )
        results, model = self.trainer.model_evaluation()
        print(f"Model report: \n{results}")
        print('==============================')
        print('Training Completed 🚀')
        save_model(
            model= model,
            dir_path=r'E:\python\pytorch\Models'
        )


    def run(self):
        # Single function to setup everything and train
        self.setup()
        self.train()

if __name__=="__main__":
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    obj = FoodImageClassifier(
        url=url,
        image_size=64,
        hidden_units=10,
        batch_size=32,
        learning_rate=0.01,
        epochs=2,
        device=device
    )

    obj.run()