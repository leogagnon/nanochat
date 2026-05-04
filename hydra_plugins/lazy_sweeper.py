"""
Hydra sweeper that defers OmegaConf resolver evaluation to each job process.

BasicSweeper resolves the params config in the main (orchestrator) process, so
custom resolvers like ${uniform:0.1,1.0} are evaluated once and every job receives
the same value.  This subclass overrides _parse_config to read the raw (unresolved)
strings instead, so each job resolves the interpolation independently.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class LazyBasicSweeperConf:
    _target_: str = "hydra_plugins.lazy_sweeper.LazyBasicSweeper"
    max_batch_size: Optional[int] = None
    params: Optional[Dict[str, str]] = None


ConfigStore.instance().store(
    group="hydra/sweeper",
    name="lazy_basic",
    node=LazyBasicSweeperConf,
    provider="nanochat",
)


class LazyBasicSweeper(BasicSweeper):
    def _parse_config(self) -> List[str]:
        if self.config.hydra.sweeper.params is None:
            return []
        raw = OmegaConf.to_container(self.config.hydra.sweeper.params, resolve=False)
        return [f"{k}={v}" for k, v in raw.items()]
