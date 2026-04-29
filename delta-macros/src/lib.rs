use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

/// Derive macro for the `Module` trait.
///
/// Automatically implements:
/// - `module_name()` → returns the struct's name as a `String`
/// - `parameters()` → collects all fields of type `nn::Parameter`
/// - `submodules()` → collects all fields that implement `Module`
///   (detected via the `#[module]` field attribute)
///
/// The user must still implement `forward()` manually.
///
/// # Field Attributes
///
/// - `#[module]`  — marks a field as a sub-module (must be `Box<dyn Module>` or
///   a concrete type implementing `Module`).  The macro will include it in the
///   `submodules()` return value.
///
/// Fields of type `nn::Parameter` are picked up **automatically** without any
/// attribute, because the type itself is unambiguous.
///
/// # Example
///
/// ```ignore
/// use your_crate::nn;
/// use module_derive::Module;
///
/// #[derive(Module)]
/// pub struct Linear {
///     pub weight: nn::Parameter,
///     pub bias: nn::Parameter,
///     #[module]
///     pub activation: Box<dyn Module>,
/// }
///
/// impl Module for Linear {
///     fn forward(&self, args: Vec<IValue>, kwargs: HashMap<String, IValue>) -> IValue {
///         todo!()
///     }
/// }
/// ```
#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(
                    struct_name,
                    "Module derive does not support tuple structs",
                )
                .to_compile_error()
                .into()
            }
            Fields::Unit => {
                // Unit struct — no parameters or submodules
                return expand(struct_name, &struct_name_str, vec![], vec![]);
            }
        },
        _ => {
            return syn::Error::new_spanned(struct_name, "Module can only be derived for structs")
                .to_compile_error()
                .into()
        }
    };

    let mut param_fields = Vec::new();
    let mut submodule_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().expect("named field");

        // Check for #[module] attribute → submodule
        let is_submodule = field
            .attrs
            .iter()
            .any(|a| a.path().is_ident("module"));

        if is_submodule {
            submodule_fields.push(field_name.clone());
        } else if is_nn_parameter(&field.ty) {
            // Automatically detect nn::Parameter fields
            param_fields.push(field_name.clone());
        }
    }

    expand(struct_name, &struct_name_str, param_fields, submodule_fields)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` when the type path ends in `Parameter` (covers both
/// `nn::Parameter` and a bare `Parameter` import).
fn is_nn_parameter(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(last) = type_path.path.segments.last() {
            return last.ident == "Parameter";
        }
    }
    false
}

/// Emit the `module_name`, `parameters`, and `submodules` impl block.
fn expand(
    struct_name: &syn::Ident,
    struct_name_str: &str,
    param_fields: Vec<syn::Ident>,
    submodule_fields: Vec<syn::Ident>,
) -> TokenStream {
    let parameters_body = if param_fields.is_empty() {
        quote! { vec![] }
    } else {
        quote! {
            vec![
                #( self.#param_fields.clone() ),*
            ]
        }
    };

    let submodules_body = if submodule_fields.is_empty() {
        quote! { vec![] }
    } else {
        quote! {
            vec![
                #( &self.#submodule_fields as &dyn Module ),*
            ]
        }
    };

    quote! {
        impl Module for #struct_name {
            fn module_name(&self) -> String {
                #struct_name_str.to_string()
            }

            fn parameters(&self) -> Vec<nn::Parameter> {
                #parameters_body
            }

            fn submodules(&self) -> Vec<&dyn Module> {
                #submodules_body
            }
        }
    }
    .into()
}