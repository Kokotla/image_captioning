import torch
from PIL import Image
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import pickle
from captionModel import EncoderDecoder, EncoderCNN, DecoderRNN, Attention
from preprocess import Vocabulary
import argparse

def save_image(inp, title=None):
    """Imshow for Tensor."""
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks

    # Save the image using matplotlib.pyplot.imsave()
    plt.savefig('./res/sample_image.png')
    plt.pause(0.001)


def get_caps_from(img, model, vocabs):
    # generate the caption

    features_tensors = transformation_effv2(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=vocabs)
        caption = ' '.join(caps)
        save_image(img, title=caption)
    return caption


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Please give the path for the image you want a caption for")
    args = parser.parse_args()

    model_weight = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    transformation_effv2 = model_weight.transforms()

    img = Image.open(args.image).convert("RGB")

    model = torch.load("captioning_model.pt").to(device)

    with open('captioning_vocab.pkl', 'rb') as f:
        vocabs = pickle.load(f)

    cap = get_caps_from(img, model, vocabs)
    print(cap)

