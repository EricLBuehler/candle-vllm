// src/tools/schema.rs
//! JSON Schema utilities for tool parameters
//!
//! Provides helpers for working with JSON Schema in tool definitions.

use crate::tools::Tool;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Builder for creating JSON Schema objects
#[derive(Debug, Clone, Default)]
pub struct SchemaBuilder {
    schema_type: String,
    properties: HashMap<String, Value>,
    required: Vec<String>,
    description: Option<String>,
    additional_properties: Option<bool>,
}

impl SchemaBuilder {
    /// Create a new object schema builder
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
            description: None,
            additional_properties: None,
        }
    }

    /// Add a description to the schema
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a string property
    pub fn string_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "string",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a number property
    pub fn number_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "number",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an integer property
    pub fn integer_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "integer",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a boolean property
    pub fn boolean_prop(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "boolean",
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an array property
    pub fn array_prop(
        mut self,
        name: impl Into<String>,
        items_type: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "array",
                "items": { "type": items_type.into() },
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add an enum property
    pub fn enum_prop(
        mut self,
        name: impl Into<String>,
        values: Vec<&str>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        self.properties.insert(
            name.clone(),
            json!({
                "type": "string",
                "enum": values,
                "description": description.into()
            }),
        );
        if required {
            self.required.push(name);
        }
        self
    }

    /// Add a custom property with full schema
    pub fn custom_prop(mut self, name: impl Into<String>, schema: Value, required: bool) -> Self {
        let name = name.into();
        self.properties.insert(name.clone(), schema);
        if required {
            self.required.push(name);
        }
        self
    }

    /// Disallow additional properties
    pub fn no_additional_properties(mut self) -> Self {
        self.additional_properties = Some(false);
        self
    }

    /// Build the final JSON Schema
    pub fn build(self) -> Value {
        let mut schema = json!({
            "type": self.schema_type,
            "properties": self.properties,
            "required": self.required
        });

        if let Some(desc) = self.description {
            schema["description"] = json!(desc);
        }

        if let Some(additional) = self.additional_properties {
            schema["additionalProperties"] = json!(additional);
        }

        schema
    }
}

/// Common tool schemas for built-in tools
pub mod common {
    use super::*;

    /// Calculator tool schema
    pub fn calculator_schema() -> Value {
        SchemaBuilder::object()
            .description("Evaluate a mathematical expression")
            .string_prop(
                "expression",
                "The mathematical expression to evaluate",
                true,
            )
            .build()
    }

    /// Web search tool schema
    pub fn web_search_schema() -> Value {
        SchemaBuilder::object()
            .description("Search the web for information")
            .string_prop("query", "The search query", true)
            .integer_prop("max_results", "Maximum number of results to return", false)
            .build()
    }

    /// Get current time tool schema
    pub fn get_time_schema() -> Value {
        SchemaBuilder::object()
            .description("Get current date and time")
            .string_prop(
                "timezone",
                "Timezone (e.g., 'UTC', 'America/New_York')",
                false,
            )
            .build()
    }

    /// Code execution tool schema
    pub fn code_execution_schema() -> Value {
        SchemaBuilder::object()
            .description("Execute code in a sandboxed environment")
            .enum_prop(
                "language",
                vec!["python", "javascript", "rust"],
                "Programming language",
                true,
            )
            .string_prop("code", "The code to execute", true)
            .build()
    }
}

/// Convert a Value schema to a Vec of Tool objects using ToolBuilder
/// The schema should be an object where keys are tool names and values are tool schemas
pub fn schema_to_tools(schema: &Value) -> Vec<Tool> {
    let mut tools = Vec::new();
    if let Value::Object(obj) = schema {
        for (name, tool_schema) in obj {
            if let Value::Object(props) = tool_schema {
                if let Some(params) = props.get("parameters") {
                    let builder = crate::tools::ToolBuilder::new(name.clone(), "".to_string())
                        .parameters_schema(params.clone());
                    tools.push(builder.build());
                }
            }
        }
    }
    tools
}
