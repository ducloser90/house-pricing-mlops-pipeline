"""
Request schema for the house price prediction API.

Mirrors the raw Ames Housing feature set as it exists BEFORE any of the
training-time feature engineering (column dropping, ordinal encoding, log
transform). The API replicates that engineering internally before calling
the trained sklearn Pipeline.

Field nullability matches how the training data treats missingness:
- Most fields are required (a missing value here is a real data problem).
- A specific set of categorical fields legitimately use NaN/None to mean
  "this house does not have this feature" (no garage, no pool, no fence,
  no alley access, no misc feature, no basement, no fireplace). These are
  Optional[str] = None, matching how OrdinalEncoding.apply_transformation
  fillna("None") and the ColumnTransformer's most_frequent imputer treat
  them during training.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    # --- Zoning / lot ---------------------------------------------------
    MS_SubClass: int = Field(..., alias="MS SubClass")
    MS_Zoning: str = Field(..., alias="MS Zoning")
    Lot_Frontage: float = Field(..., alias="Lot Frontage")
    Lot_Area: float = Field(..., alias="Lot Area")
    Street: str
    Alley: Optional[str] = None  # NaN = no alley access
    Lot_Shape: str = Field(..., alias="Lot Shape")
    Land_Contour: str = Field(..., alias="Land Contour")
    Utilities: str
    Lot_Config: str = Field(..., alias="Lot Config")
    Land_Slope: str = Field(..., alias="Land Slope")
    Neighborhood: str
    Condition_1: str = Field(..., alias="Condition 1")
    Condition_2: str = Field(..., alias="Condition 2")
    Bldg_Type: str = Field(..., alias="Bldg Type")
    House_Style: str = Field(..., alias="House Style")

    # --- Overall quality / age -------------------------------------------
    Overall_Qual: int = Field(..., alias="Overall Qual")
    Overall_Cond: int = Field(..., alias="Overall Cond")
    Year_Built: int = Field(..., alias="Year Built")
    Year_Remod_Add: int = Field(..., alias="Year Remod/Add")
    Roof_Style: str = Field(..., alias="Roof Style")
    Roof_Matl: str = Field(..., alias="Roof Matl")
    Exterior_1st: str = Field(..., alias="Exterior 1st")
    Exterior_2nd: str = Field(..., alias="Exterior 2nd")
    Mas_Vnr_Type: Optional[str] = Field(None, alias="Mas Vnr Type")  # NaN = no masonry veneer
    Mas_Vnr_Area: float = Field(..., alias="Mas Vnr Area")
    Exter_Qual: str = Field(..., alias="Exter Qual")
    Exter_Cond: str = Field(..., alias="Exter Cond")
    Foundation: str

    # --- Basement ----------------------------------------------------------
    Bsmt_Qual: Optional[str] = Field(None, alias="Bsmt Qual")          # NaN = no basement
    Bsmt_Cond: Optional[str] = Field(None, alias="Bsmt Cond")          # NaN = no basement
    Bsmt_Exposure: Optional[str] = Field(None, alias="Bsmt Exposure")  # NaN = no basement
    BsmtFin_Type_1: Optional[str] = Field(None, alias="BsmtFin Type 1")  # NaN = no basement
    BsmtFin_SF_1: float = Field(..., alias="BsmtFin SF 1")
    BsmtFin_Type_2: Optional[str] = Field(None, alias="BsmtFin Type 2")  # NaN = no basement
    BsmtFin_SF_2: float = Field(..., alias="BsmtFin SF 2")
    Bsmt_Unf_SF: float = Field(..., alias="Bsmt Unf SF")
    Total_Bsmt_SF: float = Field(..., alias="Total Bsmt SF")
    Bsmt_Full_Bath: float = Field(..., alias="Bsmt Full Bath")
    Bsmt_Half_Bath: float = Field(..., alias="Bsmt Half Bath")

    # --- Heating / utilities -----------------------------------------------
    Heating: str
    Heating_QC: str = Field(..., alias="Heating QC")
    Central_Air: str = Field(..., alias="Central Air")
    Electrical: str

    # --- Living area ---------------------------------------------------------
    First_Flr_SF: float = Field(..., alias="1st Flr SF")
    Second_Flr_SF: float = Field(..., alias="2nd Flr SF")
    Low_Qual_Fin_SF: float = Field(..., alias="Low Qual Fin SF")
    Gr_Liv_Area: float = Field(..., alias="Gr Liv Area")
    Full_Bath: int = Field(..., alias="Full Bath")
    Half_Bath: int = Field(..., alias="Half Bath")
    Bedroom_AbvGr: int = Field(..., alias="Bedroom AbvGr")
    Kitchen_AbvGr: int = Field(..., alias="Kitchen AbvGr")
    Kitchen_Qual: str = Field(..., alias="Kitchen Qual")
    TotRms_AbvGrd: int = Field(..., alias="TotRms AbvGrd")
    Functional: str

    # --- Fireplace -----------------------------------------------------------
    Fireplaces: int
    Fireplace_Qu: Optional[str] = Field(None, alias="Fireplace Qu")  # NaN = no fireplace

    # --- Garage --------------------------------------------------------------
    Garage_Type: Optional[str] = Field(None, alias="Garage Type")    # NaN = no garage
    Garage_Yr_Blt: Optional[float] = Field(None, alias="Garage Yr Blt")  # NaN = no garage
    Garage_Finish: Optional[str] = Field(None, alias="Garage Finish")  # NaN = no garage
    Garage_Cars: float = Field(..., alias="Garage Cars")
    Garage_Area: float = Field(..., alias="Garage Area")
    Garage_Qual: Optional[str] = Field(None, alias="Garage Qual")    # NaN = no garage
    Garage_Cond: Optional[str] = Field(None, alias="Garage Cond")    # NaN = no garage
    Paved_Drive: str = Field(..., alias="Paved Drive")

    # --- Outdoor / misc --------------------------------------------------------
    Wood_Deck_SF: float = Field(..., alias="Wood Deck SF")
    Open_Porch_SF: float = Field(..., alias="Open Porch SF")
    Enclosed_Porch: float = Field(..., alias="Enclosed Porch")
    Three_Ssn_Porch: float = Field(..., alias="3Ssn Porch")
    Screen_Porch: float = Field(..., alias="Screen Porch")
    Pool_Area: float = Field(..., alias="Pool Area")
    Pool_QC: Optional[str] = Field(None, alias="Pool QC")            # NaN = no pool
    Fence: Optional[str] = None                                      # NaN = no fence
    Misc_Feature: Optional[str] = Field(None, alias="Misc Feature")  # NaN = no misc feature
    Misc_Val: float = Field(..., alias="Misc Val")
    Mo_Sold: int = Field(..., alias="Mo Sold")
    Yr_Sold: int = Field(..., alias="Yr Sold")
    Sale_Type: str = Field(..., alias="Sale Type")
    Sale_Condition: str = Field(..., alias="Sale Condition")

    class Config:
        populate_by_name = True  # allow both the Python attr name and the alias
        json_schema_extra = {
            "example": {
                "MS SubClass": 60,
                "MS Zoning": "RL",
                "Lot Frontage": 65.0,
                "Lot Area": 8450,
                "Street": "Pave",
                "Alley": None,
                "Lot Shape": "Reg",
                "Land Contour": "Lvl",
                "Utilities": "AllPub",
                "Lot Config": "Inside",
                "Land Slope": "Gtl",
                "Neighborhood": "CollgCr",
                "Condition 1": "Norm",
                "Condition 2": "Norm",
                "Bldg Type": "1Fam",
                "House Style": "2Story",
                "Overall Qual": 7,
                "Overall Cond": 5,
                "Year Built": 2003,
                "Year Remod/Add": 2003,
                "Roof Style": "Gable",
                "Roof Matl": "CompShg",
                "Exterior 1st": "VinylSd",
                "Exterior 2nd": "VinylSd",
                "Mas Vnr Type": "BrkFace",
                "Mas Vnr Area": 196.0,
                "Exter Qual": "Gd",
                "Exter Cond": "TA",
                "Foundation": "PConc",
                "Bsmt Qual": "Gd",
                "Bsmt Cond": "TA",
                "Bsmt Exposure": "No",
                "BsmtFin Type 1": "GLQ",
                "BsmtFin SF 1": 706.0,
                "BsmtFin Type 2": "Unf",
                "BsmtFin SF 2": 0.0,
                "Bsmt Unf SF": 150.0,
                "Total Bsmt SF": 856.0,
                "Bsmt Full Bath": 1,
                "Bsmt Half Bath": 0,
                "Heating": "GasA",
                "Heating QC": "Ex",
                "Central Air": "Y",
                "Electrical": "SBrkr",
                "1st Flr SF": 856.0,
                "2nd Flr SF": 854.0,
                "Low Qual Fin SF": 0.0,
                "Gr Liv Area": 1710.0,
                "Full Bath": 2,
                "Half Bath": 1,
                "Bedroom AbvGr": 3,
                "Kitchen AbvGr": 1,
                "Kitchen Qual": "Gd",
                "TotRms AbvGrd": 8,
                "Functional": "Typ",
                "Fireplaces": 1,
                "Fireplace Qu": "TA",
                "Garage Type": "Attchd",
                "Garage Yr Blt": 2003.0,
                "Garage Finish": "RFn",
                "Garage Cars": 2.0,
                "Garage Area": 548.0,
                "Garage Qual": "TA",
                "Garage Cond": "TA",
                "Paved Drive": "Y",
                "Wood Deck SF": 0.0,
                "Open Porch SF": 61.0,
                "Enclosed Porch": 0.0,
                "3Ssn Porch": 0.0,
                "Screen Porch": 0.0,
                "Pool Area": 0.0,
                "Pool QC": None,
                "Fence": None,
                "Misc Feature": None,
                "Misc Val": 0.0,
                "Mo Sold": 2,
                "Yr Sold": 2008,
                "Sale Type": "WD ",
                "Sale Condition": "Normal",
            }
        }


class PredictionResponse(BaseModel):
    predicted_sale_price: float
    log_predicted_sale_price: float
    model_version: Optional[str] = None