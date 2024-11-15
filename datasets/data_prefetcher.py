import torch


def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [
        {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
    ]
    return samples, targets


class data_prefetcher:
    def __init__(self, loader, device, prefetch=True):
        """
        The purpose of this class is to preload data from a loader object
          and move it to the device (GPU) for faster processing.
        """
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except (
            StopIteration
        ):  # 如果使用Dataloader的next()方法取完了所有元素，再次调用next()方法会抛出StopIteration异常
            self.next_samples = None
            self.next_targets = None
            return
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(
                self.next_samples, self.next_targets, self.device
            )

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
