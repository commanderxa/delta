use crate::Tensor;

#[derive(Debug)]
pub enum IValue {
    Tensor(Tensor),
    TensorList(Vec<Tensor>),
    Tuple(Vec<IValue>),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    None,
}

impl IValue {
    /// Unwraps the IValue as a Tensor, panics if it is not a Tensor variant
    pub fn unwrap_tensor(self) -> Tensor {
        match self {
            IValue::Tensor(t) => t,
            other => panic!("expected Tensor, got {:?}", other),
        }
    }

    /// Unwraps the IValue as a tuple (Vec<IValue>), panics otherwise
    pub fn unwrap_tuple(self) -> Vec<IValue> {
        match self {
            IValue::Tuple(v) => v,
            other => panic!("expected Tuple, got {:?}", other),
        }
    }

    /// Unwraps the Tensor as a tensor list (Vec<Tensor>), panics otherwise
    pub fn unwrap_list(self) -> Vec<Tensor> {
        match self {
            IValue::TensorList(v) => v,
            other => panic!("expected Tuple, got {:?}", other),
        }
    }

    /// Unwraps the IValue as i64, panics otherwise
    pub fn unwrap_int(self) -> i64 {
        match self {
            IValue::Int(i) => i,
            other => panic!("expected Int, got {:?}", other),
        }
    }

    /// Unwraps the IValue as f64, panics otherwise
    pub fn unwrap_float(self) -> f64 {
        match self {
            IValue::Float(f) => f,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    /// Unwraps the IValue as bool, panics otherwise
    pub fn unwrap_bool(self) -> bool {
        match self {
            IValue::Bool(b) => b,
            other => panic!("expected Bool, got {:?}", other),
        }
    }

    /// Unwraps the IValue as String, panics otherwise
    pub fn unwrap_str(self) -> String {
        match self {
            IValue::Str(s) => s,
            other => panic!("expected Str, got {:?}", other),
        }
    }
}

/// Creates a tuple of `(Vec<IValue>, HashMap<String, IValue>)` from positional args and keyword args.
///
/// # Syntax
/// ```ignore
/// ivalue![[arg1, arg2, ...], { key1: val1, key2: val2, ... }]
/// ```
///
/// - The first bracket `[...]` contains positional arguments.
/// - The second brace `{...}` contains keyword arguments as `key: value` pairs.
/// - All values are automatically converted via `IValue::from(...)`,
///   so any type implementing `From<T> for IValue` works directly.
///
/// # Shorthand Forms
/// - `ivalue![[x, y]]`         — args only, empty kwargs
/// - `ivalue![{ key: val }]`   — kwargs only, empty args
/// - `ivalue![[], {}]`         — fully empty
///
/// # Examples
/// ```ignore
/// # use delta::ivalue;
/// # use delta::Tensor;
/// # use delta::nn;
/// # use delta::nn::Module;
/// # let x = Tensor::zeros(&[1, 1]);
/// # let q = Tensor::zeros(&[1, 1]);
/// # let k = Tensor::zeros(&[1, 1]);
/// # let v = Tensor::zeros(&[1, 1]);
///
/// // args only shorthand
/// let (args, kwargs) = ivalue![[x]];
///
/// // kwargs only shorthand
/// let (args, kwargs) = ivalue![{ mask: true, scale: 1.0f64 }];
///
/// // multiple positional args
/// let (args, kwargs) = ivalue![[q, k, v]];
///
/// // both args and kwargs
/// let (args, kwargs) = ivalue![[x], { dropout: false }];
///
/// // inline directly into forward call
/// let model = nn::Linear::new(1, 1, true);
/// let (args, kwargs) = ivalue![[x], { scale: 0.5f64 }];
/// model.forward(args, kwargs);
///
/// // fully empty
/// let (args, kwargs) = ivalue![[], {}];
/// ```
#[macro_export]
macro_rules! ivalue {
    // fully empty: ivalue![[], {}]
    ([], {}) => {
        (vec![], std::collections::HashMap::new())
    };

    // args only shorthand: ivalue![[x, y]]
    // sugar for ivalue![[x, y], {}]
    ([$($arg:expr),* $(,)?]) => {
        $crate::ivalue!([$($arg),*], {})
    };

    // kwargs only shorthand: ivalue![{ key: val }]
    // sugar for ivalue![[], { key: val }]
    ({ $($key:ident : $val:expr),* $(,)? }) => {
        $crate::ivalue!([], { $($key : $val),* })
    };

    // args only, explicit empty kwargs: ivalue![[x, y], {}]
    ([$($arg:expr),* $(,)?], {}) => {
        (
            // wrap each positional arg into IValue via From trait
            vec![$($crate::ivalue::IValue::from($arg)),*],
            std::collections::HashMap::new(),
        )
    };

    // kwargs only, explicit empty args: ivalue![[], { key: val }]
    ([], { $($key:ident : $val:expr),* $(,)? }) => {
        (
            vec![],
            {
                let mut map = std::collections::HashMap::new();
                // stringify the identifier key and wrap value into IValue via From trait
                $(map.insert(
                    stringify!($key).to_string(),
                    $crate::ivalue::IValue::from($val)
                );)*
                map
            }
        )
    };

    // both args and kwargs: ivalue![[x], { mask: true }]
    ([$($arg:expr),* $(,)?], { $($key:ident : $val:expr),* $(,)? }) => {
        (
            // positional args → Vec<IValue>
            vec![$($crate::ivalue::IValue::from($arg)),*],
            {
                // keyword args → HashMap<String, IValue>
                let mut map = std::collections::HashMap::new();
                $(map.insert(
                    stringify!($key).to_string(),
                    $crate::ivalue::IValue::from($val)
                );)*
                map
            }
        )
    };
}

impl From<Tensor> for IValue {
    fn from(t: Tensor) -> Self {
        IValue::Tensor(t)
    }
}

impl From<Vec<Tensor>> for IValue {
    fn from(v: Vec<Tensor>) -> Self {
        IValue::TensorList(v)
    }
}

impl From<i64> for IValue {
    fn from(i: i64) -> Self {
        IValue::Int(i)
    }
}

impl From<f64> for IValue {
    fn from(f: f64) -> Self {
        IValue::Float(f)
    }
}

impl From<bool> for IValue {
    fn from(b: bool) -> Self {
        IValue::Bool(b)
    }
}

impl From<String> for IValue {
    fn from(s: String) -> Self {
        IValue::Str(s)
    }
}

impl From<&str> for IValue {
    fn from(s: &str) -> Self {
        IValue::Str(s.to_owned())
    }
}
