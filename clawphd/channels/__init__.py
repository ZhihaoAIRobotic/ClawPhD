"""Chat channels module with plugin architecture."""

from clawphd.channels.base import BaseChannel
from clawphd.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
