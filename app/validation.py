"""Input validation for Streamlit app."""

class InputValidator:
    """Validate user inputs."""
    
    CONSTRAINTS = {
        'BHK': {'min': 1, 'max': 10},
        'Size_in_SqFt': {'min': 300, 'max': 50000},
        'Price_in_Lakhs': {'min': 10, 'max': 5000},
        'Year_Built': {'min': 1950, 'max': 2025},
        'Floor_No': {'min': 0, 'max': 100},
        'Total_Floors': {'min': 1, 'max': 100},
        'Nearby_Schools': {'min': 0, 'max': 30},
        'Nearby_Hospitals': {'min': 0, 'max': 30},
    }
    
    @staticmethod
    def validate(bhk, size, price, year, floor, total_floors, schools, hospitals):
        """Validate all inputs."""
        errors = []
        
        if not 1 <= bhk <= 10:
            errors.append("BHK must be 1-10")
        if not 300 <= size <= 50000:
            errors.append("Size must be 300-50000 SqFt")
        if not 10 <= price <= 5000:
            errors.append("Price must be 10-5000 Lakhs")
        if not 1950 <= year <= 2025:
            errors.append("Year must be 1950-2025")
        if not 0 <= floor <= 100:
            errors.append("Floor must be 0-100")
        if not 1 <= total_floors <= 100:
            errors.append("Total Floors must be 1-100")
        if floor > total_floors:
            errors.append("Floor cannot exceed Total Floors")
        if size == 0:
            errors.append("Size cannot be zero")
        if price <= 0:
            errors.append("Price must be positive")
        
        return len(errors) == 0, errors
