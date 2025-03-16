def epoch_time(start_time, end_time):
    """
    calculate the elapsed time
    Args:
        start_time: start time
        end_time: end time
    Returns:
        (int, int): elapsed minutes, elapsed seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs