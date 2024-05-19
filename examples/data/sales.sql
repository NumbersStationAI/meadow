-- Creating the 'customers' table
CREATE TABLE customers (
    customer_id INTEGER,
    name VARCHAR,
    email VARCHAR,
    state VARCHAR,
    city VARCHAR,
    zip_code VARCHAR
);

-- Inserting sample data into 'customers'
INSERT INTO customers (customer_id, name, email, state, city, zip_code) VALUES
(1, 'Alice Brown', 'alice.brown@example.com', 'CA', 'Los Angeles', '90001'),
(2, 'Bob Smith', 'bob.smith@example.com', 'NY', 'New York', '10001'),
(3, 'Charlie Johnson', 'charlie.johnson@example.com', 'TX', 'Houston', '77001'),
(4, 'David Lee', 'david.lee@example.com', 'FL', 'Miami', '33101'),
(5, 'Eva White', 'eva.white@example.com', 'IL', 'Chicago', '60601'),
(6, 'Frank Green', 'frank.green@example.com', 'PA', 'Philadelphia', '19019'),
(7, 'Grace Hall', 'grace.hall@example.com', 'OH', 'Columbus', '43215'),
(8, 'Hannah Young', 'hannah.young@example.com', 'WA', 'Seattle', '98101'),
(9, 'Ian Walker', 'ian.walker@example.com', 'GA', 'Atlanta', '30301'),
(10, 'Julia King', 'julia.king@example.com', 'MA', 'Boston', '02101'),
(11, 'Karl Adams', 'karl.adams@example.com', 'AZ', 'Phoenix', '85001'),
(12, 'Laura Brown', 'laura.brown@example.com', 'CO', 'Denver', '80201'),
(13, 'Mason Ray', 'mason.ray@example.com', 'MI', 'Detroit', '48201'),
(14, 'Nina Morris', 'nina.morris@example.com', 'WA', 'Spokane', '99201'),
(15, 'Oscar Knight', 'oscar.knight@example.com', 'NC', 'Charlotte', '28201'),
(16, 'Pamela Hart', 'pamela.hart@example.com', 'NV', 'Las Vegas', '89101'),
(17, 'Quinn Lee', 'quinn.lee@example.com', 'OR', 'Portland', '97201'),
(18, 'Rita Patel', 'rita.patel@example.com', 'TX', 'Austin', '73301'),
(19, 'Samuel Teague', 'samuel.teague@example.com', 'MN', 'Minneapolis', '55401'),
(20, 'Tina Mason', 'tina.mason@example.com', 'LA', 'New Orleans', '70112');


-- Creating the 'demographics' table
CREATE TABLE demographics (
    customer_id INTEGER,  -- Unique identifier for a customer, should match the customer_id in the customers table
    age INTEGER,  -- Age of the customer
    gender VARCHAR(10), -- Gender of the customer
    income DECIMAL(10, 2),  -- Approximate annual income of the customer
    education VARCHAR(50),  -- Highest level of education achieved by the customer
    occupation VARCHAR(50),  -- Current occupation of the customer
    marital_status VARCHAR(20),  -- Marital status of the customer
    household_size INTEGER,  -- Number of people in the customer's household
    PRIMARY KEY (customer_id)  -- Ensures each customer_id is unique within this table
);

-- Inserting sample data into 'customers'
INSERT INTO demographics (customer_id, age, gender, income, education, occupation, marital_status, household_size) VALUES
(1, 28, 'Female', 55000.00, 'Bachelor', 'Software Developer', 'Single', 1),
(2, 34, 'Male', 72000.00, 'Master', 'Graphic Designer', 'Married', 3),
(3, 45, 'Male', 95000.00, 'PhD', 'Project Manager', 'Married', 4),
(4, 22, 'Male', 34000.00, 'Bachelor', 'Sales Associate', 'Single', 2),
(5, 37, 'Female', 63000.00, 'Master', 'HR Manager', 'Divorced', 1),
(6, 29, 'Male', 49000.00, 'Bachelor', 'Teacher', 'Single', 1),
(7, 42, 'Female', 88000.00, 'PhD', 'Research Scientist', 'Married', 3),
(8, 55, 'Female', 120000.00, 'PhD', 'Consultant', 'Widowed', 2),
(9, 26, 'Male', 47000.00, 'Bachelor', 'Marketing Coordinator', 'Single', 2),
(10, 33, 'Female', 76000.00, 'Master', 'Physician', 'Married', 3),
(11, 48, 'Male', 101000.00, 'PhD', 'Engineer', 'Divorced', 1),
(12, 30, 'Female', 69000.00, 'Bachelor', 'Dentist', 'Single', 1),
(13, 25, 'Male', 54000.00, 'Bachelor', 'Architect', 'Single', 2),
(14, 39, 'Female', 58000.00, 'Master', 'School Principal', 'Married', 4),
(15, 51, 'Male', 83000.00, 'Master', 'Lawyer', 'Married', 4),
(16, 36, 'Female', 77000.00, 'PhD', 'Business Analyst', 'Single', 1),
(17, 44, 'Non-binary', 67000.00, 'Master', 'Journalist', 'Divorced', 2),
(18, 53, 'Female', 102000.00, 'PhD', 'Pharmacist', 'Married', 3),
(19, 27, 'Male', 56000.00, 'Bachelor', 'Civil Engineer', 'Single', 1),
(20, 31, 'Female', 81000.00, 'Master', 'Accountant', 'Married', 3);


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
INSERT INTO suppliers (supplier_id, name, contact_name, address, city, state, zip_code) VALUES
(1, 'Quality Goods', 'Tom Hardy', '1234 Pine St', 'Los Angeles', 'CA', '90001'),
(2, 'Tech Supplies', 'Nina Morris', '2345 Oak St', 'New York', 'NY', '10001'),
(3, 'Paper Company', 'Dave Lee', '3456 Maple St', 'Dallas', 'TX', '75001'),
(4, 'Office Furnishings', 'Samuel Teague', '4567 Elm St', 'Miami', 'FL', '33101'),
(5, 'Industrial Tools', 'Liz Bright', '5678 Spruce St', 'Chicago', 'IL', '60601'),
(6, 'Food Suppliers', 'Anne Marie', '6789 Cedar St', 'Philadelphia', 'PA', '19101'),
(7, 'Medical Supplies', 'John Carter', '7890 Birch St', 'Cleveland', 'OH', '44101'),
(8, 'Clothing Materials', 'Clara Oswald', '8901 Dogwood St', 'Seattle', 'WA', '98101'),
(9, 'Automotive Parts', 'Bruce Wayne', '9012 Fir St', 'Atlanta', 'GA', '30301'),
(10, 'Beauty Supplies', 'Diana Prince', '0123 Redwood St', 'Boston', 'MA', '02101'),
(11, 'Sporting Goods', 'Leah Thompson', '1357 Willow St', 'San Francisco', 'CA', '94102'),
(12, 'Electronics Depot', 'Max Turner', '2468 Ivy St', 'Austin', 'TX', '73301'),
(13, 'Farm Fresh', 'Olivia Grant', '3579 Poplar St', 'Portland', 'OR', '97201'),
(14, 'Bookstore Supplies', 'Ethan Hunt', '4680 Linden St', 'Orlando', 'FL', '32801'),
(15, 'Toy Factory', 'Mia Kang', '5791 Ash St', 'Denver', 'CO', '80201'),
(16, 'Music Instruments', 'Noah Lee', '6802 Oak St', 'Nashville', 'TN', '37201'),
(17, 'Gardening Supplies', 'Lily Evans', '7913 Pine St', 'Phoenix', 'AZ', '85001'),
(18, 'Art Supplies', 'James Potter', '8924 Maple St', 'Albuquerque', 'NM', '87101'),
(19, 'Pet Supplies', 'Katie Bell', '9035 Spruce St', 'St. Louis', 'MO', '63101'),
(20, 'Furniture Factory', 'Logan Wright', '0146 Elm St', 'Raleigh', 'NC', '27601');


-- Creating the 'products' table
CREATE TABLE products (
    product_id INTEGER,
    name VARCHAR,
    category VARCHAR,  -- Sample categoires are Office Supplies, Pantry, Electronics, etc.
    description VARCHAR,
    price DECIMAL(10, 2),  -- Recorded stable price of product that may change over time
    supplier_id INTEGER
);

-- Inserting sample data into 'products'
INSERT INTO products (product_id, name, category, description, price, supplier_id) VALUES
(1, 'Laptop', 'Electronics', 'High performance laptop for gaming and professional use', 1200.00, 2),
(2, 'Desktop Computer', 'Electronics', 'Reliable and powerful office computer', 800.00, 2),
(3, 'Printer', 'Office Supplies', 'Efficient and fast printing', 150.00, 2),
(4, 'Office Chair', 'Office Furniture', 'Ergonomic office chair for maximum comfort', 300.00, 4),
(5, 'Desk', 'Office Furniture', 'Spacious office desk with modern design', 450.00, 4),
(6, 'Notebook', 'Office Supplies', '100 pages notebook, spiral bound', 3.00, 3),
(7, 'Pen Set', 'Office Supplies', 'High-quality pens for smooth writing', 25.00, 3),
(8, 'Monitor', 'Electronics', '27-inch full HD monitor, great for productivity', 230.00, 2),
(9, 'Keyboard', 'Electronics', 'Mechanical keyboard with backlit keys', 70.00, 2),
(10, 'Mouse', 'Electronics', 'Wireless mouse, ergonomic design', 25.00, 2),
(11, 'Smartphone', 'Electronics', 'Latest model with advanced features', 999.00, 12),
(12, 'Tablet', 'Electronics', 'Portable and powerful tablet for on-the-go use', 600.00, 12),
(13, 'Headphones', 'Electronics', 'Noise-canceling headphones', 199.00, 12),
(14, 'Camera', 'Electronics', 'Digital camera with high resolution', 450.00, 12),
(15, 'Charger', 'Electronics', 'Fast charger compatible with multiple devices', 20.00, 12),
(16, 'Backpack', 'Accessories', 'Durable backpack with laptop compartment', 60.00, 11),
(17, 'Water Bottle', 'Accessories', 'Insulated water bottle to keep drinks cold or hot', 30.00, 11),
(18, 'Notebook', 'Office Supplies', 'Eco-friendly notebook made from recycled materials', 10.00, 13),
(19, 'Calendar', 'Office Supplies', '2024 desk calendar', 12.00, 13),
(20, 'Binder', 'Office Supplies', 'Heavy-duty binder for organizing papers', 8.00, 13);


-- Creating the 'orders' table
CREATE TABLE orders (
    order_id INTEGER,
    customer_id INTEGER,
    order_date DATE
);

-- Inserting sample data into 'orders'
INSERT INTO orders (order_id, customer_id, order_date) VALUES
(1, 1, '2023-04-01'),
(2, 1, '2023-04-03'),
(3, 2, '2023-04-05'),
(4, 2, '2023-04-07'),
(5, 2, '2023-04-09'),
(6, 3, '2023-04-11'),
(7, 4, '2023-04-13'),
(8, 4, '2023-04-15'),
(9, 5, '2023-04-17'),
(10, 5, '2023-04-19'),
(11, 6, '2023-04-21'),
(12, 6, '2023-04-23'),
(13, 7, '2023-04-25'),
(14, 7, '2023-04-27'),
(15, 8, '2023-04-29'),
(16, 8, '2023-05-01'),
(17, 9, '2023-05-03'),
(18, 9, '2023-05-05'),
(19, 10, '2023-05-07'),
(20, 10, '2023-05-09'),
(21, 11, '2023-05-11'),
(22, 11, '2023-05-13'),
(23, 12, '2023-05-15'),
(24, 12, '2023-05-17'),
(25, 13, '2023-05-19'),
(26, 13, '2023-05-21'),
(27, 14, '2023-05-23'),
(28, 14, '2023-05-25'),
(29, 15, '2023-05-27'),
(30, 15, '2023-05-29');


-- Creating the 'order_items' table
CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10, 2)  -- The price per unit at the time of the order, may be different due to promotions or sales
);

-- Inserting sample data into 'order_items'
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1200.00),
(1, 3, 2, 140.00),
(2, 2, 1, 790.00),
(2, 4, 1, 280.00),
(3, 5, 1, 440.00),
(3, 6, 3, 2.50),
(3, 4, 3, 280.00),
(4, 7, 5, 22.00),
(4, 9, 2, 65.00),
(5, 10, 1, 24.00),
(5, 8, 1, 220.00),
(6, 11, 1, 950.00),
(6, 13, 1, 180.00),
(7, 14, 1, 430.00),
(7, 15, 3, 18.00),
(8, 16, 2, 55.00),
(8, 17, 1, 28.00),
(9, 18, 4, 9.00),
(9, 19, 2, 11.00),
(10, 20, 3, 7.50),
(10, 6, 10, 2.80),
(11, 1, 1, 1190.00),
(11, 3, 2, 135.00),
(12, 2, 1, 800.00),
(12, 4, 1, 299.00),
(13, 5, 2, 450.00),
(13, 6, 1, 3.00),
(14, 7, 2, 25.00),
(14, 9, 1, 70.00),
(15, 10, 1, 25.00),
(15, 8, 1, 230.00),
(16, 11, 2, 999.00),
(16, 13, 1, 199.00),
(17, 14, 2, 450.00),
(17, 15, 1, 20.00),
(18, 16, 1, 60.00),
(18, 17, 2, 30.00),
(19, 18, 1, 10.00),
(19, 19, 2, 12.00),
(20, 20, 4, 8.00),
(20, 6, 5, 3.00),
(21, 1, 1, 1200.00),
(21, 3, 2, 150.00),
(22, 2, 1, 800.00),
(22, 4, 1, 300.00),
(23, 5, 1, 450.00),
(23, 6, 2, 3.00),
(24, 7, 3, 25.00),
(24, 9, 1, 70.00),
(25, 10, 1, 25.00),
(25, 8, 1, 230.00),
(26, 11, 1, 950.00),
(26, 13, 1, 190.00),
(27, 14, 1, 450.00),
(27, 15, 2, 20.00),
(28, 16, 1, 60.00),
(28, 17, 1, 30.00),
(29, 18, 2, 10.00),
(29, 19, 3, 12.00),
(30, 20, 5, 8.00),
(30, 6, 15, 3.00);
