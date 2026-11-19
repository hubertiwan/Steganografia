class AppError(Exception):
    """Error shown to the user: what went wrong and how to fix it."""

    def __init__(self, message, hint=""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self):
        return f"{self.message}\n{self.hint}" if self.hint else self.message


class Cancelled(AppError):
    def __init__(self, done=None):
        super().__init__("Przerwano na prośbę użytkownika.")
        self.done = done  # number of finished prompts, if known
