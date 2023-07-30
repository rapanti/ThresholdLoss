class MultiCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, num_crops):
        self.base_transform = base_transform
        self.num_crops = num_crops

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.num_crops)]
