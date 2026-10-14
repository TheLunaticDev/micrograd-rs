use std::cell::RefCell;

fn main() {
    let arena = Arena::new();
    let a = arena.insert(-4.0);
    let b = arena.insert(2.0);
    println!("{:.4}", a.data().unwrap());
    println!("{:.4}", b.data().unwrap());
}

/// Struct that holds the value.
struct Node {
    data: f64,
}

/// A sharable object that holds nodes and allows operations.
struct Arena {
    nodes: RefCell<Vec<Node>>,
}

impl Arena {
    fn new() -> Self {
        Arena { nodes: Vec::<Node>::new().into() }
    }

    fn insert(&self, val: f64) -> Value<'_> {
        let node = Node { data: val };
        let mut borrow = self.nodes.borrow_mut();
        borrow.push(node);
        let idx = borrow.len() - 1;
        Value {
            idx,
            arena: self,
        }
    }
}

#[derive(Clone, Copy)]
struct Value<'a> {
    idx: usize,
    arena: &'a Arena,
}

impl Value<'_> {
    fn data(&self) -> Option<f64> {
        self.arena.nodes.borrow().get(self.idx).map(|val| val.data)
    }
}
