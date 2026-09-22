class DomainError(Exception):
    """Ожидаемая ошибка приложения, которую транспорт переводит в свой формат."""


class TicketNotFound(DomainError):
    pass


class TicketConflict(DomainError):
    pass


class ModelUnavailable(DomainError):
    pass


class DatabaseUnavailable(DomainError):
    pass


class InferenceUnavailable(DomainError):
    pass


class InferenceTimeout(InferenceUnavailable):
    pass


class MalformedLLMOutput(DomainError):
    pass
