from abc import ABC, abstractmethod

import jaxtyping as jx
import numpy as np


class BaseTransform(ABC):
    @property
    @abstractmethod
    def in_feature_name(self) -> str:
        """The name of the input feature that this transform expects.

        Returns:
            str: The name of the input feature.
        """

    @property
    @abstractmethod
    def out_feature_name(self) -> str:
        """The name of the output feature that this transform produces.

        Returns:
            str: The name of the output feature.
        """

    @abstractmethod
    def __call__(
        self,
        feature: jx.Float[np.ndarray, "events jets"],
    ) -> jx.Float[np.ndarray, "events jets"]:
        """App Args:
            feature (Float[Array, "events"]): The input feature to transform.

        Returns:
            Float[Array, "events"]: The transformed feature.
        """

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return (
            f"{self.__class__.__name__}("
            f"in_feature_name={self.in_feature_name}, "
            f"out_feature_name={self.out_feature_name})"
        )


class IdentityTransform(BaseTransform):
    """A transform that returns the input feature unchanged."""

    def __init__(
        self,
        in_feature_name: str,
    ):
        """Initialize the IdentityTransform with the input feature name.

        Args:
            in_feature_name (str): The name of the input feature.
        """
        self._in_feature_name = in_feature_name
        self._out_feature_name = in_feature_name

    @property
    def in_feature_name(self) -> str:
        return self._in_feature_name

    @property
    def out_feature_name(self) -> str:
        return self._out_feature_name

    def __call__(
        self,
        feature: jx.Float[np.ndarray, "events jets"],
    ) -> jx.Float[np.ndarray, "events jets"]:
        """Return the input feature unchanged.

        Args:
            feature (Float[Array, "events"]): The input feature to transform.

        Returns:
            Float[Array, "events"]: The unchanged input feature.
        """
        return feature


class LogTransform(BaseTransform):
    """A transform that applies the natural logarithm to the input feature."""

    def __init__(
        self,
        in_feature_name: str,
        epsilon: float = 1e-10,
    ):
        """Initialize the LogTransform with the input feature name.

        Args:
            in_feature_name (str): The name of the input feature.
            epsilon (float): A small value to avoid log(0) issues.
                Defaults to 1e-10.
        """
        self._in_feature_name = in_feature_name
        self._epsilon = epsilon

    @property
    def in_feature_name(self) -> str:
        return self._in_feature_name

    @property
    def out_feature_name(self) -> str:
        return f"log_{self._in_feature_name}"

    def __call__(
        self,
        feature: jx.Float[np.ndarray, "events jets"],
    ) -> jx.Float[np.ndarray, "events jets"]:
        """Apply the natural logarithm to the input feature.

        Args:
            feature (Float[Array, "events"]): The input feature to transform.

        Returns:
            Float[Array, "events"]: The transformed feature after applying log.
        """
        return np.log(feature + self._epsilon)


class ScaledLogTransform(BaseTransform):
    """A transform that applies a scaled natural logarithm to the input feature."""

    def __init__(
        self,
        in_feature_name: str,
        scale: float = 1.0,
        offset: float = 0.0,
        epsilon: float = 1e-10,
    ):
        """Initialize the ScaledLogTransform with the input feature name and scale.

        Args:
            in_feature_name (str): The name of the input feature.
            scale (float): The scaling factor for the log transformation.
                Defaults to 1.0.
            offset (float): The offset to apply after the log transformation.
            epsilon (float): A small value to avoid log(0) issues.
                Defaults to 1e-10.
        """
        self._in_feature_name = in_feature_name
        self._scale = scale
        self._offset = offset
        self._epsilon = epsilon

    @property
    def in_feature_name(self) -> str:
        return self._in_feature_name

    @property
    def out_feature_name(self) -> str:
        return f"scaled_log_{self._in_feature_name}"

    def __call__(
        self,
        feature: jx.Float[np.ndarray, "events jets"],
    ) -> jx.Float[np.ndarray, "events jets"]:
        """Apply a scaled natural logarithm to the input feature.

        Args:
            feature (Float[Array, "events"]): The input feature to transform.

        Returns:
            Float[Array, "events"]: The transformed feature after applying scaled log.
        """
        return (np.log(feature + self._epsilon) - self._offset) / self._scale


class CosTransform(BaseTransform):
    """A transform that applies the cosine function to the input feature."""

    def __init__(
        self,
        in_feature_name: str,
    ):
        """Initialize the CosTransform with the input feature name.

        Args:
            in_feature_name (str): The name of the input feature.
        """
        self._in_feature_name = in_feature_name

    @property
    def in_feature_name(self) -> str:
        return self._in_feature_name

    @property
    def out_feature_name(self) -> str:
        return f"cos_{self._in_feature_name}"

    def __call__(
        self,
        feature: jx.Float[np.ndarray, "events jets"],
    ) -> jx.Float[np.ndarray, "events jets"]:
        """Apply the cosine function to the input feature.

        Args:
            feature (Float[Array, "events"]): The input feature to transform.

        Returns:
            Float[Array, "events"]: The transformed feature after applying cosine.
        """
        return np.cos(feature)


class SinTransform(BaseTransform):
    """A transform that applies the sine function to the input feature."""

    def __init__(
        self,
        in_feature_name: str,
    ):
        """Initialize the SinTransform with the input feature name.

        Args:
            in_feature_name (str): The name of the input feature.
                Used to query the input feature from the h5 file.
        """
        self._in_feature_name = in_feature_name

    @property
    def in_feature_name(self) -> str:
        return self._in_feature_name

    @property
    def out_feature_name(self) -> str:
        return f"sin_{self._in_feature_name}"

    def __call__(
        self,
        feature: jx.Float[np.ndarray, "events jets"],
    ) -> jx.Float[np.ndarray, "events jets"]:
        """Apply the sine function to the input feature.

        Args:
            feature (Float[Array, "events"]): The input feature to transform.

        Returns:
            Float[Array, "events"]: The transformed feature after applying sine.
        """
        return np.sin(feature)
