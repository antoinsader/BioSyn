import os, time,   psutil, torch


class MetricsLogger:
    def __init__(self, logger, confs, tag="train"):
        self.use_cuda = confs.use_cuda
        self.device = "cuda" if self.use_cuda else "cpu"
        self.logger = logger
        self.tag = tag
        self.process = psutil.Process(os.getpid())
        
        self.cpu_memory_used = 0.0
        self.messages = []
        self.one_time_events_set = set()


    def current_cpu_mem_usage(self):
        rss = self.process.memory_info().rss / (1024 * 2)
        self.cpu_memory_used = rss
        return rss


    def current_gpu_mem_usage(self):
        if self.use_cuda:
            free = torch.cuda.mem_get_info()[0] / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return (free, total)
        return (0.0,0.0)


    def current_gpu_stats(self):
        """
            alloc (current allocated memory in MB): Memory currently allocated by tensors.
            alloc_peak (peak allocated memory in MB): Highest memory allocated by tensors since the program start or last reset.
            res (current reserved memory in MB): Memory reserved by the caching allocator (includes allocated plus cached blocks).
            res_peak (peak reserved memory in MB): Highest reserved memory since the program start or last reset.

        """
        if not self.use_cuda:
            return (None, None, None, None)
        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device) / (1024**2)
        alloc_peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
        res = torch.cuda.memory_reserved(self.device) / (1024**2)
        res_peak = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        return (alloc, alloc_peak, res, res_peak)



    def log_event(self, event_tag, t0=None, log_immediate=True, first_iteration_only=False, only_elapsed_time=False):
        if first_iteration_only and event_tag in self.one_time_events_set:
            return True


        self.one_time_events_set.add(event_tag)
        msg = f"[{self.tag}-{event_tag}] "


        if t0:
            elapsed = time.time() - t0
            msg += f" | elapsed time: {elapsed:.5f}seconds "


        if only_elapsed_time:
            return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")

        msg += f" | CPU Memory usage: {self.current_cpu_mem_usage():.1f}MB "
        if self.use_cuda:
            (free, total) = self.current_gpu_mem_usage()
            msg += f" | GPU memory total/free: {total:.1f}/{free:.1f}MB"
            (alloc, alloc_peak, res, res_peak) = self.current_gpu_stats()
            msg += f" | CUDA: allocated/peak: {alloc:.1f}/{alloc_peak:.1f}MB, reserved/peak {res:.1f}{res_peak:.1f}MB"

        return self.logger.info(f"\n{msg}") if log_immediate else self.messages.append(f"\n{msg}")


