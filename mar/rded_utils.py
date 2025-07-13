import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def el2n_score(pred, y):
    with torch.no_grad():
        pred = F.softmax(pred, dim=1)
        l2_loss = torch.nn.MSELoss(reduction='none')
        y_hot = F.one_hot(y, num_classes=pred.shape[1])
        el2n = torch.sqrt(l2_loss(y_hot, pred).sum(dim=1))
    return el2n.cpu()

# use 0 to pad "other three picture"
def pad(input_tensor, target_height, target_width=None):
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)
    horizontal_padding = target_width - input_tensor.size(3)

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    padded_tensor = F.pad(
        input_tensor, (left_padding, right_padding, top_padding, bottom_padding)
    )

    return padded_tensor


def batched_forward(model, tensor, batch_size):
    total_samples = tensor.size(0)

    all_outputs = []

    model.eval()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i : min(i + batch_size, total_samples)]

            output = model(batch_data)

            all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)

    return final_output


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=224, factor=2):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

denormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def selector(n, model, images, labels, size, m=5):
    with torch.no_grad():
        # [mipc, m, 3, 224, 224]
        images = images.cuda()
        s = images.shape

        # # [mipc * m, 3, 224, 224]
        images = images.permute(1, 0, 2, 3, 4)
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])

        # [mipc * m, 1]
        labels = labels.repeat(m).cuda()

        # [mipc * m, n_class]
        batch_size = s[0]  # Change it for small GPU memory
        preds = batched_forward(model, pad(images, size).cuda(), batch_size)

        # [mipc * m]
        dist = cross_entropy(preds, labels)

        # [m, mipc]
        dist = dist.reshape(m, s[0])

        # [mipc]
        index = torch.argmin(dist, 0)
        dist = dist[index, torch.arange(s[0])]

        # [mipc, 3, 224, 224]
        sa = images.shape
        images = images.reshape(m, s[0], sa[1], sa[2], sa[3])
        images = images[index, torch.arange(s[0])]

    indices = torch.argsort(dist, descending=False)[:n]
    torch.cuda.empty_cache()
    return images[indices].detach()


def selector_el2n(n, model, images, labels, size, m=5):
    with torch.no_grad():
        # [mipc, m, 3, 224, 224]
        images = images.cuda()
        s = images.shape

        # # [mipc * m, 3, 224, 224]
        images = images.permute(1, 0, 2, 3, 4)
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])

        # [mipc * m, 1]
        labels = labels.repeat(m).cuda()

        # [mipc * m, n_class]
        batch_size = s[0]  # Change it for small GPU memory
        preds = batched_forward(model, pad(images, size).cuda(), batch_size)

        # [mipc * m]
        dist = el2n_score(preds, labels)

        # [m, mipc]
        dist = dist.reshape(m, s[0])

        # [mipc]
        index = torch.argmin(dist, 0)
        dist = dist[index, torch.arange(s[0])]

        # [mipc, 3, 224, 224]
        sa = images.shape
        images = images.reshape(m, s[0], sa[1], sa[2], sa[3])
        images = images[index, torch.arange(s[0])]

    indices = torch.argsort(dist, descending=False)[:n]
    torch.cuda.empty_cache()
    return images[indices].detach()


def mix_images(input_img, out_size, factor, n):
    s = out_size // factor
    remained = out_size % factor
    k = 0
    mixed_images = torch.zeros(
        (n, 3, out_size, out_size),
        requires_grad=False,
        dtype=torch.float,
    )
    h_loc = 0
    for i in range(factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(factor):
            w_r = s + 1 if j < remained else s
            img_part = F.interpolate(
                input_img.data[k * n : (k + 1) * n], size=(h_r, w_r)
            )
            mixed_images.data[
                0:n,
                :,
                h_loc : h_loc + h_r,
                w_loc : w_loc + w_r,
            ] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images