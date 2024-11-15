"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""

from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
from datasets.pycocotools import COCO


class CocoDetection(VisionDataset):

    def __init__(
        self,
        root,
        annFile,
        args,
        cls_order,
        phase_idx,
        incremental,
        incremental_val,
        val_each_phase,
        balanced_ft,
        tfs_or_tfh,
        num_of_phases,
        cls_per_phase,
        seed_data,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        local_rank=0,
        local_size=1,
        is_rehearsal=False,
        is_train=False,
    ):
        super(CocoDetection, self).__init__(
            root, transforms, transform, target_transform
        )
        self.coco = COCO(
            args,
            cls_order,
            phase_idx,
            incremental,
            incremental_val,
            val_each_phase,
            balanced_ft,
            tfs_or_tfh,
            num_of_phases,
            cls_per_phase,
            seed_data,
            annFile,
            is_rehearsal=is_rehearsal,
            is_train=is_train,
        )
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]["file_name"]
            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert("RGB")
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
