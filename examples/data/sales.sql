-- Creating the 'customers' table
CREATE TABLE customers (
    customer_id INTEGER,
    name VARCHAR,
    email VARCHAR,
    state VARCHAR,
    zip_code VARCHAR
);

-- Inserting sample data into 'customers'
INSERT INTO customers VALUES
(1, 'Alice Brown', 'alice.brown@example.com', 'CA', '90001'),
(2, 'Bob Smith', 'bob.smith@example.com', 'NY', '10001'),
(3, 'Charlie Johnson', 'charlie.johnson@example.com', 'TX', '75001'),
(4, 'David Lee', 'david.lee@example.com', 'FL', '33001'),
(5, 'Eva White', 'eva.white@example.com', 'IL', '60001'),
(6, 'Frank Green', 'frank.green@example.com', 'PA', '19001'),
(7, 'Grace Hall', 'grace.hall@example.com', 'OH', '43001'),
(8, 'Hannah Young', 'hannah.young@example.com', 'WA', '98001'),
(9, 'Ian Walker', 'ian.walker@example.com', 'GA', '30001'),
(10, 'Julia King', 'julia.king@example.com', 'MA', '02001');

-- Creating the 'suppliers' table
CREATE TABLE suppliers (
    supplier_id INTEGER,
    name VARCHAR,
    contact_name VARCHAR,
    address VARCHAR,
    city VARCHAR,
    state VARCHAR,
    zip_code VARCHAR
);

-- Inserting sample data into 'suppliers'
INSERT INTO suppliers VALUES
(1, 'Quality Goods', 'Tom Hardy', '1234 Pine St', 'Los Angeles', 'CA', '90001'),
(2, 'Tech Supplies', 'Nina Morris', '2345 Oak St', 'New York', 'NY', '10001'),
(3, 'Paper Company', 'Dave Lee', '3456 Maple St', 'Dallas', 'TX', '75001'),
(4, 'Office Furnishings', 'Samuel Teague', '4567 Elm St', 'Miami', 'FL', '33101'),
(5, 'Industrial Tools', 'Liz Bright', '5678 Spruce St', 'Chicago', 'IL', '60601'),
(6, 'Food Suppliers', 'Anne Marie', '6789 Cedar St', 'Philadelphia', 'PA', '19101'),
(7, 'Medical Supplies', 'John Carter', '7890 Birch St', 'Cleveland', 'OH', '44101'),
(8, 'Clothing Materials', 'Clara Oswald', '8901 Dogwood St', 'Seattle', 'WA', '98101'),
(9, 'Automotive Parts', 'Bruce Wayne', '9012 Fir St', 'Atlanta', 'GA', '30301'),
(10, 'Beauty Supplies', 'Diana Prince', '0123 Redwood St', 'Boston', 'MA', '02101');

-- Creating the 'products' table
CREATE TABLE products (
    product_id INTEGER,
    name VARCHAR,
    category VARCHAR,
    description VARCHAR,
    price DECIMAL(10, 2),
    supplier_id INTEGER
);

-- Inserting sample data into 'products'
INSERT INTO products VALUES
(1, 'Laptop', 'Office Supplies', 'High performance laptop', 1200.00, 2),
(2, 'Desktop Computer', 'Office Supplies', 'Reliable and powerful office computer', 800.00, 2),
(3, 'Printer', 'Office Supplies', 'Efficient and fast printing', 150.00, 2),
(4, 'Office Chair', 'Office Supplies', 'Ergonomic office chair', 300.00, 4),
(5, 'Desk', 'Office Supplies', 'Spacious office desk', 450.00, 4),
(6, 'Notebook', 'Office Supplies', '100 pages notebook', 3.00, 3),
(7, 'Apple', 'Pantry', 'Apple fruit', 1.50, 3),
(8, 'Pear', 'Pantry', 'Pear frut', 230.00, 2),
(9, 'Chips', 'Pantry', 'Lays regular chips in bag', 70.00, 1),
(10, 'Cookies', 'Pantry', 'Nestles chocolate chip cookies', 25.00, 2);

-- Creating the 'orders' table
CREATE TABLE orders (
    order_id INTEGER,
    customer_id INTEGER,
    order_date DATE,
    total DECIMAL(10, 2)
);

-- Inserting sample data into 'orders'
INSERT INTO orders VALUES
(1, 1, '2023-01-10', 1250.00),
(2, 2, '2023-01-11', 830.00),
(3, 3, '2023-01-12', 153.00),
(4, 4, '2023-01-13', 305.00),
(5, 5, '2023-01-14', 453.00),
(6, 6, '2023-01-15', 5.50),
(7, 7, '2023-01-16', 2.50),
(8, 8, '2023-01-17', 232.00),
(9, 9, '2023-01-18', 71.00),
(10, 10, '2023-01-19', 27.00);

-- Creating the 'order_items' table
CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER
);

-- Inserting sample data into 'order_items'
INSERT INTO order_items VALUES
(1, 1, 1),
(2, 2, 1),
(3, 3, 4),
(4, 4, 1),
(5, 5, 10),
(6, 6, 1),
(7, 7, 3),
(8, 8, 2),
(9, 9, -1),
(10, 10, 1);