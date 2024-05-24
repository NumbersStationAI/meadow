-- customers JOIN demographics ON customers.cust_num = demographics.cid
-- suppliers JOIN products ON suppliers.supp_id = products.supplier_alt_id
-- orders JOIN order_items ON orders.order_num = order_items.ord_ref
-- orders JOIN products ON CAST(products.prod_id AS text) = order_items.prod_num
-- orders JOIN customers ON orders.cust_ref = customers.cust_num

-- Creating the 'customers' table
CREATE TABLE customers (
    alt_id VARCHAR,
    cust_num INTEGER,
    name VARCHAR,
    email VARCHAR,
    state VARCHAR,
    city VARCHAR,
    zip_code VARCHAR
);

-- Inserting sample data into 'customers'
INSERT INTO customers (alt_id, cust_num, name, email, state, city, zip_code) VALUES
('abc123', 1001, 'Alice Brown', 'alice.brown@example.com', 'CA', 'Los Angeles', '90001'),
('def456', 1002, 'Bob Smith', 'bob.smith@example.com', 'NY', 'New York', '10001'),
('ghi789', 1003, 'Charlie Johnson', 'charlie.johnson@example.com', 'TX', 'Houston', '77001'),
('jkl012', 1004, 'David Lee', 'david.lee@example.com', 'FL', 'Miami', '33101'),
('mno345', 1005, 'Eva White', 'eva.white@example.com', 'IL', 'Chicago', '60601'),
('pqr678', 1006, 'Frank Green', 'frank.green@example.com', 'PA', 'Philadelphia', '19019'),
('stu901', 1007, 'Grace Hall', 'grace.hall@example.com', 'OH', 'Columbus', '43215'),
('vwx234', 1008, 'Hannah Young', 'hannah.young@example.com', 'WA', 'Seattle', '98101'),
('yza567', 1009, 'Ian Walker', 'ian.walker@example.com', 'GA', 'Atlanta', '30301'),
('bcd890', 1010, 'Julia King', 'julia.king@example.com', 'MA', 'Boston', '02101'),
('efg123', 1011, 'Karl Adams', 'karl.adams@example.com', 'AZ', 'Phoenix', '85001'),
('hij456', 1012, 'Laura Brown', 'laura.brown@example.com', 'CO', 'Denver', '80201'),
('klm789', 1013, 'Mason Ray', 'mason.ray@example.com', 'MI', 'Detroit', '48201'),
('nop012', 1014, 'Nina Morris', 'nina.morris@example.com', 'WA', 'Spokane', '99201'),
('qrs345', 1015, 'Oscar Knight', 'oscar.knight@example.com', 'NC', 'Charlotte', '28201'),
('tuv678', 1016, 'Pamela Hart', 'pamela.hart@example.com', 'NV', 'Las Vegas', '89101'),
('wxy901', 1017, 'Quinn Lee', 'quinn.lee@example.com', 'OR', 'Portland', '97201'),
('zab234', 1018, 'Rita Patel', 'rita.patel@example.com', 'TX', 'Austin', '73301'),
('cde567', 1019, 'Samuel Teague', 'samuel.teague@example.com', 'MN', 'Minneapolis', '55401'),
('fgh890', 1020, 'Tina Mason', 'tina.mason@example.com', 'LA', 'New Orleans', '70112');

-- Creating the 'demographics' table
CREATE TABLE demographics (
    cid INTEGER,
    did INTEGER,
    cust_num VARCHAR,
    age INTEGER,
    gender VARCHAR(10),
    income DECIMAL(10, 2),
    education VARCHAR(50),
    occupation VARCHAR(50),
    marital_status VARCHAR(20),
    household_size INTEGER,
    PRIMARY KEY (did)
);

-- Inserting sample data into 'demographics'
INSERT INTO demographics (cid, did, cust_num, age, gender, income, education, occupation, marital_status, household_size) VALUES
(1001, 1, 'abc', 28, 'Female', 55000.00, 'Bachelor', 'Software Developer', 'Single', 1),
(1002, 2, 'def', 34, 'Male', 72000.00, 'Master', 'Graphic Designer', 'Married', 3),
(1003, 3, 'ghi', 45, 'Male', 95000.00, 'PhD', 'Project Manager', 'Married', 4),
(1004, 4, 'jkl', 22, 'Male', 34000.00, 'Bachelor', 'Sales Associate', 'Single', 2),
(1005, 5, 'mno', 37, 'Female', 63000.00, 'Master', 'HR Manager', 'Divorced', 1),
(1006, 6, 'pqr', 29, 'Male', 49000.00, 'Bachelor', 'Teacher', 'Single', 1),
(1007, 7, 'stu', 42, 'Female', 88000.00, 'PhD', 'Research Scientist', 'Married', 3),
(1008, 8, 'vwx', 55, 'Female', 120000.00, 'PhD', 'Consultant', 'Widowed', 2),
(1009, 9, 'yza', 26, 'Male', 47000.00, 'Bachelor', 'Marketing Coordinator', 'Single', 2),
(1010, 10, '890', 33, 'Female', 76000.00, 'Master', 'Physician', 'Married', 3),
(1011, 11, '123', 48, 'Male', 101000.00, 'PhD', 'Engineer', 'Divorced', 1),
(1012, 12, 'ij4', 30, 'Female', 69000.00, 'Bachelor', 'Dentist', 'Single', 1),
(1013, 13, 'm78', 25, 'Male', 54000.00, 'Bachelor', 'Architect', 'Single', 2),
(1014, 14, '012', 39, 'Female', 58000.00, 'Master', 'School Principal', 'Married', 4),
(1015, 15, 'q45', 51, 'Male', 83000.00, 'Master', 'Lawyer', 'Married', 4),
(1016, 16, 't78', 36, 'Female', 77000.00, 'PhD', 'Business Analyst', 'Single', 1),
(1017, 17, 'w01', 44, 'Non-binary', 67000.00, 'Master', 'Journalist', 'Divorced', 2),
(1018, 18, '234', 53, 'Female', 102000.00, 'PhD', 'Pharmacist', 'Married', 3),
(1019, 19, 'cde', 27, 'Male', 56000.00, 'Bachelor', 'Civil Engineer', 'Single', 1),
(1020, 20, '890', 31, 'Female', 81000.00, 'Master', 'Accountant', 'Married', 3);

-- Creating the 'suppliers' table
CREATE TABLE suppliers (
    supp_id VARCHAR,
    alt_id INTEGER,
    name VARCHAR,
    contact_name VARCHAR,
    address VARCHAR,
    city VARCHAR,
    state VARCHAR,
    zip_code VARCHAR
);

-- Inserting sample data into 'suppliers'
INSERT INTO suppliers (supp_id, alt_id, name, contact_name, address, city, state, zip_code) VALUES
('S1', 1001, 'Quality Goods', 'Tom Hardy', '1234 Pine St', 'Los Angeles', 'CA', '90001'),
('S2', 1002, 'Tech Supplies', 'Nina Morris', '2345 Oak St', 'New York', 'NY', '10001'),
('S3', 1003, 'Paper Company', 'Dave Lee', '3456 Maple St', 'Dallas', 'TX', '75001'),
('S4', 1004, 'Office Furnishings', 'Samuel Teague', '4567 Elm St', 'Miami', 'FL', '33101'),
('S5', 1005, 'Industrial Tools', 'Liz Bright', '5678 Spruce St', 'Chicago', 'IL', '60601'),
('S6', 1006, 'Food Suppliers', 'Anne Marie', '6789 Cedar St', 'Philadelphia', 'PA', '19101'),
('S7', 1007, 'Medical Supplies', 'John Carter', '7890 Birch St', 'Cleveland', 'OH', '44101'),
('S8', 1008, 'Clothing Materials', 'Clara Oswald', '8901 Dogwood St', 'Seattle', 'WA', '98101'),
('S9', 1009, 'Automotive Parts', 'Bruce Wayne', '9012 Fir St', 'Atlanta', 'GA', '30301'),
('S10', 1010, 'Beauty Supplies', 'Diana Prince', '0123 Redwood St', 'Boston', 'MA', '02101'),
('S11', 1011, 'Sporting Goods', 'Leah Thompson', '1357 Willow St', 'San Francisco', 'CA', '94102'),
('S12', 1012, 'Electronics Depot', 'Max Turner', '2468 Ivy St', 'Austin', 'TX', '73301'),
('S13', 1013, 'Farm Fresh', 'Olivia Grant', '3579 Poplar St', 'Portland', 'OR', '97201'),
('S14', 1014, 'Bookstore Supplies', 'Ethan Hunt', '4680 Linden St', 'Orlando', 'FL', '32801'),
('S15', 1015, 'Toy Factory', 'Mia Kang', '5791 Ash St', 'Denver', 'CO', '80201'),
('S16', 1016, 'Music Instruments', 'Noah Lee', '6802 Oak St', 'Nashville', 'TN', '37201'),
('S17', 1017, 'Gardening Supplies', 'Lily Evans', '7913 Pine St', 'Phoenix', 'AZ', '85001'),
('S18', 1018, 'Art Supplies', 'James Potter', '8924 Maple St', 'Albuquerque', 'NM', '87101'),
('S19', 1019, 'Pet Supplies', 'Katie Bell', '9035 Spruce St', 'St. Louis', 'MO', '63101'),
('S20', 1020, 'Furniture Factory', 'Logan Wright', '0146 Elm St', 'Raleigh', 'NC', '27601');

-- Creating the 'products' table
CREATE TABLE products (
    prod_id INTEGER,
    name VARCHAR,
    category VARCHAR,
    description VARCHAR,
    price DECIMAL(10, 2),
    supplier_alt_id VARCHAR
);

-- Inserting sample data into 'products'
INSERT INTO products (prod_id, name, category, description, price, supplier_alt_id) VALUES
(1001, 'Laptop', 'Electronics', 'High performance laptop for gaming and professional use', 1200.00, 'S2'),
(1002, 'Desktop Computer', 'Electronics', 'Reliable and powerful office computer', 800.00, 'S2'),
(1003, 'Printer', 'Office Supplies', 'Efficient and fast printing', 150.00, 'S2'),
(1004, 'Office Chair', 'Office Furniture', 'Ergonomic office chair for maximum comfort', 300.00, 'S4'),
(1005, 'Desk', 'Office Furniture', 'Spacious office desk with modern design', 450.00, 'S4'),
(1006, 'Notebook', 'Office Supplies', '100 pages notebook, spiral bound', 3.00, 'S3'),
(1007, 'Pen Set', 'Office Supplies', 'High-quality pens for smooth writing', 25.00, 'S3'),
(1008, 'Monitor', 'Electronics', '27-inch full HD monitor, great for productivity', 230.00, 'S2'),
(1009, 'Keyboard', 'Electronics', 'Mechanical keyboard with backlit keys', 70.00, 'S2'),
(1010, 'Mouse', 'Electronics', 'Wireless mouse, ergonomic design', 25.00, 'S2'),
(1011, 'Smartphone', 'Electronics', 'Latest model with advanced features', 999.00, 'S12'),
(1012, 'Tablet', 'Electronics', 'Portable and powerful tablet for on-the-go use', 600.00, 'S12'),
(1013, 'Headphones', 'Electronics', 'Noise-canceling headphones', 199.00, 'S12'),
(1014, 'Camera', 'Electronics', 'Digital camera with high resolution', 450.00, 'S12'),
(1015, 'Charger', 'Electronics', 'Fast charger compatible with multiple devices', 20.00, 'S12'),
(1016, 'Backpack', 'Accessories', 'Durable backpack with laptop compartment', 60.00, 'S11'),
(1017, 'Water Bottle', 'Accessories', 'Insulated water bottle to keep drinks cold or hot', 30.00, 'S11'),
(1018, 'Notebook', 'Office Supplies', 'Eco-friendly notebook made from recycled materials', 10.00, 'S13'),
(1019, 'Calendar', 'Office Supplies', '2024 desk calendar', 12.00, 'S13'),
(1020, 'Binder', 'Office Supplies', 'Heavy-duty binder for organizing papers', 8.00, 'S13');

-- Creating the 'orders' table
CREATE TABLE orders (
    order_num VARCHAR,
    order_id INTEGER,
    cust_ref INTEGER,
    order_date DATE
);

-- Inserting sample data into 'orders'
INSERT INTO orders (order_num, order_id, cust_ref, order_date) VALUES
('ORD1001', 1, 1001, '2023-04-01'),
('ORD1002', 2, 1001, '2023-04-03'),
('ORD1003', 3, 1002, '2023-04-05'),
('ORD1004', 4, 1002, '2023-04-07'),
('ORD1005', 5, 1002, '2023-04-09'),
('ORD1006', 6, 1003, '2023-04-11'),
('ORD1007', 7, 1004, '2023-04-13'),
('ORD1008', 8, 1004, '2023-04-15'),
('ORD1009', 9, 1005, '2023-04-17'),
('ORD1010', 10, 1005, '2023-04-19'),
('ORD1011', 11, 1006, '2023-04-21'),
('ORD1012', 12, 1006, '2023-04-23'),
('ORD1013', 13, 1007, '2023-04-25'),
('ORD1014', 14, 1007, '2023-04-27'),
('ORD1015', 15, 1008, '2023-04-29'),
('ORD1016', 16, 1008, '2023-05-01'),
('ORD1017', 17, 1009, '2023-05-03'),
('ORD1018', 18, 1009, '2023-05-05'),
('ORD1019', 19, 1010, '2023-05-07'),
('ORD1020', 20, 1010, '2023-05-09'),
('ORD1021', 21, 1011, '2023-05-11'),
('ORD1022', 22, 1011, '2023-05-13'),
('ORD1023', 23, 1012, '2023-05-15'),
('ORD1024', 24, 1012, '2023-05-17'),
('ORD1025', 25, 1013, '2023-05-19'),
('ORD1026', 26, 1013, '2023-05-21'),
('ORD1027', 27, 1014, '2023-05-23'),
('ORD1028', 28, 1014, '2023-05-25'),
('ORD1029', 29, 1015, '2023-05-27'),
('ORD1030', 30, 1015, '2023-05-29');

-- Creating the 'order_items' table
CREATE TABLE order_items (
    item_id INTEGER,
    ord_ref VARCHAR,
    prod_num VARCHAR,
    quantity INTEGER,
    unit_price DECIMAL(10, 2)
);

-- Inserting sample data into 'order_items'
INSERT INTO order_items (item_id, ord_ref, prod_num, quantity, unit_price) VALUES
(1, 'ORD1001', '1001', 1, 1200.00),
(2, 'ORD1001', '1003', 2, 140.00),
(3, 'ORD1002', '1002', 1, 790.00),
(4, 'ORD1002', '1004', 1, 280.00),
(5, 'ORD1003', '1005', 1, 440.00),
(6, 'ORD1003', '1006', 3, 2.50),
(7, 'ORD1003', '1004', 3, 280.00),
(8, 'ORD1004', '1007', 5, 22.00),
(9, 'ORD1004', '1009', 2, 65.00),
(10, 'ORD1005', '1010', 1, 24.00),
(11, 'ORD1005', '1008', 1, 220.00),
(12, 'ORD1006', '1011', 1, 950.00),
(13, 'ORD1006', '1013', 1, 180.00),
(14, 'ORD1007', '1014', 1, 430.00),
(15, 'ORD1007', '1015', 3, 18.00),
(16, 'ORD1008', '1016', 2, 55.00),
(17, 'ORD1008', '1017', 1, 28.00),
(18, 'ORD1009', '1018', 4, 9.00),
(19, 'ORD1009', '1019', 2, 11.00),
(20, 'ORD1010', '1020', 3, 7.50),
(21, 'ORD1010', '1006', 10, 2.80),
(22, 'ORD1011', '1001', 1, 1190.00),
(23, 'ORD1011', '1003', 2, 135.00),
(24, 'ORD1012', '1002', 1, 800.00),
(25, 'ORD1012', '1004', 1, 299.00),
(26, 'ORD1013', '1005', 2, 450.00),
(27, 'ORD1013', '1006', 1, 3.00),
(28, 'ORD1014', '1007', 2, 25.00),
(29, 'ORD1014', '1009', 1, 70.00),
(30, 'ORD1015', '1010', 1, 25.00),
(31, 'ORD1015', '1008', 1, 230.00),
(32, 'ORD1016', '1011', 2, 999.00),
(33, 'ORD1016', '1013', 1, 199.00),
(34, 'ORD1017', '1014', 2, 450.00),
(35, 'ORD1017', '1015', 1, 20.00),
(36, 'ORD1018', '1016', 1, 60.00),
(37, 'ORD1018', '1017', 2, 30.00),
(38, 'ORD1019', '1018', 1, 10.00),
(39, 'ORD1019', '1019', 2, 12.00),
(40, 'ORD1020', '1020', 4, 8.00),
(41, 'ORD1020', '1006', 5, 3.00),
(42, 'ORD1021', '1001', 1, 1200.00),
(43, 'ORD1021', '1003', 2, 150.00),
(44, 'ORD1022', '1002', 1, 800.00),
(45, 'ORD1022', '1004', 1, 300.00),
(46, 'ORD1023', '1005', 1, 450.00),
(47, 'ORD1023', '1006', 2, 3.00),
(48, 'ORD1024', '1007', 3, 25.00),
(49, 'ORD1024', '1009', 1, 70.00),
(50, 'ORD1025', '1010', 1, 25.00),
(51, 'ORD1025', '1008', 1, 230.00),
(52, 'ORD1026', '1011', 1, 950.00),
(53, 'ORD1026', '1013', 1, 190.00),
(54, 'ORD1027', '1014', 1, 450.00),
(55, 'ORD1027', '1015', 2, 20.00),
(56, 'ORD1028', '1016', 1, 60.00),
(57, 'ORD1028', '1017', 1, 30.00),
(58, 'ORD1029', '1018', 2, 10.00),
(59, 'ORD1029', '1019', 3, 12.00),
(60, 'ORD1030', '1020', 5, 8.00),
(61, 'ORD1030', '1006', 15, 3.00);
