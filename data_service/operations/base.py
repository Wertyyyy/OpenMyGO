import abc
from typing import Generic, TypeVar

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
StreamOutputType = TypeVar("StreamOutputType")


class BaseOperation(abc.ABC, Generic[InputType, OutputType]):
    @abc.abstractmethod
    async def process(self, data: InputType) -> OutputType:
        pass

    async def __call__(self, data: InputType) -> OutputType:
        return await self.process(data)


class BaseStreamOperation(abc.ABC, Generic[InputType, StreamOutputType]):
    @abc.abstractmethod
    def process(self, data: InputType) -> StreamOutputType:
        pass

    def __call__(self, data: InputType) -> StreamOutputType:
        return self.process(data)
