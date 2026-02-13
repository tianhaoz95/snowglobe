pub fn generate(name: String) -> String {
    format!("Hello, {name} :)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let name = "snowglobe".to_string();
        let result = generate(name);
        assert_eq!(result, "Hello, snowglobe :)");
    }
}
