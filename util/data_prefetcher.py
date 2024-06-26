import torch


def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
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
        ):  # If all the elements have been feteched, calling the next() method again will throw a StopIteration exception.
            self.next_samples = None
            self.next_targets = None
            return
        with torch.cuda.stream(self.stream):
            # print("self.next_samples: ", self.next_samples)
            # print("self.next_targets: ", self.next_targets)
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
                targets.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                # print("samples: ", samples)
                # print("targets: ", targets)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
