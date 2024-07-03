import torch
import torch.nn as nn

# Define the U-Net model in PyTorch
class UNet3D(nn.Module):
    def __init__(self, img_height, img_width, img_depth, img_channels, num_classes):
        super(UNet3D, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1, dropout=0.1):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            )

        def upconv_block(in_channels, out_channels, kernel_size=2, stride=2, dropout=0.1):
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.encoder1 = conv_block(img_channels, 16, dropout=0.1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = conv_block(16, 32, dropout=0.1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = conv_block(32, 64, dropout=0.2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = conv_block(64, 128, dropout=0.2)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(128, 256, dropout=0.3)

        self.upconv4 = upconv_block(256, 128, dropout=0.2)
        self.decoder4 = conv_block(256, 128, dropout=0.2)

        self.upconv3 = upconv_block(128, 64, dropout=0.2)
        self.decoder3 = conv_block(128, 64, dropout=0.2)

        self.upconv2 = upconv_block(64, 32, dropout=0.1)
        self.decoder2 = conv_block(64, 32, dropout=0.1)

        self.upconv1 = upconv_block(32, 16, dropout=0.1)
        self.decoder1 = conv_block(32, 16, dropout=0.1)

        self.output_conv = nn.Conv3d(16, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat((u4, e4), dim=1))

        u3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((u3, e3), dim=1))

        u2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((u2, e2), dim=1))

        u1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((u1, e1), dim=1))

        out = self.output_conv(d1)
        out = self.softmax(out)
        return out

# Define the training function
def train(net, trainloader, optimizer, epochs, device: str):
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


# Define the evaluation function
def evaluate(net, valloader, device: str):
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(valloader.dataset)
    print(f'Validation Loss: {loss}, Accuracy: {accuracy}')
    return loss, accuracy

