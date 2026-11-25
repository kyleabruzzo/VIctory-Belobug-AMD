//! Maximum side of newly allocated streamed textures. Zero keeps all mips
//! requested by the title.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TextureCap {
    Default,
    AdaptiveNative,
    Fixed(u64),
}

impl TextureCap {
    pub(crate) fn parse(setting: Option<&str>) -> Self {
        match setting.map(str::trim) {
            Some("adaptive-native") => Self::AdaptiveNative,
            Some(s) => s.parse().map_or(Self::Default, Self::Fixed),
            None => Self::Default,
        }
    }

    pub(crate) fn limit(self, sample: impl FnOnce() -> Budget) -> u64 {
        if let Self::Fixed(maximum) = self {
            return maximum;
        }
        let budget = sample();
        match budget.percent {
            0..70 if self == Self::AdaptiveNative && budget.valid => 0,
            0..70 => 1024,
            70..85 => 512,
            _ => 256,
        }
    }
}

/// On a failed query retain the previous pressure tier for conservative
/// allocations. Only a current, nonzero budget may admit native resolution.
#[derive(Clone, Copy, Default)]
pub(crate) struct Budget {
    percent: u64,
    valid: bool,
}

impl Budget {
    pub(crate) fn observe(&mut self, local: Option<(u64, u64)>) {
        self.valid = local.is_some_and(|(_, budget)| budget != 0);
        if let Some((used, budget)) = local {
            // Preserve the default policy's pressure calculation for zero budgets.
            self.percent = used.saturating_mul(100) / budget.max(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_native_thresholds_and_default_policy() {
        for (used, native, default) in [
            (0, 0, 1024),
            (699, 0, 1024),
            (700, 512, 512),
            (849, 512, 512),
            (850, 256, 256),
            (1100, 256, 256),
        ] {
            let mut sample = Budget::default();
            sample.observe(Some((used, 1000)));
            assert_eq!(TextureCap::AdaptiveNative.limit(|| sample), native);
            assert_eq!(TextureCap::Default.limit(|| sample), default);
        }
    }

    #[test]
    fn adaptive_native_uses_conservative_quality_without_a_valid_budget() {
        let mut sample = Budget::default();
        let cap = TextureCap::AdaptiveNative;
        assert_eq!(cap.limit(|| sample), 1024);
        sample.observe(Some((20, 100)));
        assert_eq!(cap.limit(|| sample), 0);
        sample.observe(None);
        assert_eq!(cap.limit(|| sample), 1024);
        sample.observe(Some((90, 100)));
        sample.observe(None);
        assert_eq!(cap.limit(|| sample), 256);
        sample.observe(Some((0, 0)));
        assert_eq!(cap.limit(|| sample), 1024);
        sample.observe(Some((50, 100)));
        assert_eq!(cap.limit(|| sample), 0);
    }

    #[test]
    fn numeric_limits_skip_budget_queries_and_unknown_settings_keep_the_default() {
        for (setting, expected) in [("0", 0), (" 256 ", 256), ("4096", 4096)] {
            assert_eq!(
                TextureCap::parse(Some(setting)).limit(|| panic!("fixed cap queried budget")),
                expected
            );
        }
        assert_eq!(
            TextureCap::parse(Some(" adaptive-native ")),
            TextureCap::AdaptiveNative
        );
        for setting in [None, Some("native"), Some("-1"), Some("")] {
            assert_eq!(TextureCap::parse(setting), TextureCap::Default);
        }
    }
}
