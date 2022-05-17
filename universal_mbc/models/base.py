from typing import Any, Dict, List

from torch import nn


class HashableModule(nn.Module):
    _hashable_attrs: List[str] = []

    def set_hashable_attrs(self, attrs: List[str]) -> None:
        """method which can be called in the __init__ function of descendents to set the hashable attributes"""
        self._hashable_attrs = attrs

    def valid(self) -> None:
        pass
        # if len(self._hashable_attrs) == 0:
        #     raise ValueError(f"a hashable module should have some hashable attributes, got: {self._hashable_attrs=}")

    def extra_repr(self) -> str:
        """
        take every attribute that is in hashable attributes, and add it to the extra representation.
        This will cause it to be rendered to string when calling str(module) and allow each model
        setup to have a unique md5(str(module)) hash if all of the relevant features are tracked.
        """
        self.valid()
        out = ""
        for i, attr in enumerate(self._hashable_attrs):
            if i == 0:
                out += f"{attr}={getattr(self, attr)}"
                continue
            out += f", {attr}={getattr(self, attr)}"
        return out

    def check_for_duplicate_keys(self, old: Dict[str, Any], new: Dict[str, Any]) -> None:
        for k in new.keys():
            if k in old.keys():
                raise ValueError(f"found the same key in two results keys: found ({new.keys()=}) in {old.keys()=}\n{new=}\n{old=}")

    def get_results_keys(self) -> Dict[str, Any]:
        """
        get the results keys and then recursively call this on all hashable children of this module.
        This will return all of the keys necessary for storing results unique to the model. When paired
        with a simple database like a pandas DataFrame, this should allow for querying along any hyperparameter
        or model setting dimension as necessary
        """
        self.valid()
        out: Dict[str, Any] = {}
        for n, m in self.named_modules():  # modules includes self so we don't have to add self keys later
            if isinstance(m, HashableModule):
                if "sse." in n and "sse.0" not in n:
                    # this is not ideal, but we want to avoid making extra entries in the results keys when there are multiple sse parallel modules
                    # so skip if this has multiple sse's (sse.[number]) but this is not the first sse layer (all the subsequent layers should be the same)
                    continue

                new = {f"{n}:{k}": getattr(m, k) for k in m._hashable_attrs}

                self.check_for_duplicate_keys(out, new)
                out = {**out, **new}

        return out
